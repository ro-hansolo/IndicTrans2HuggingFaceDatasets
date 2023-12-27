import re
import torch
from datasets import load_dataset,Dataset
from transformers import AutoModelForSeq2SeqLM, BitsAndBytesConfig
from IndicTransTokenizer.utils import preprocess_batch, postprocess_batch
from IndicTransTokenizer.tokenizer import IndicTransTokenizer
from tqdm import tqdm
import logging
import random
import pandas as pd
import os
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# Constants
en_indic_ckpt_dir = "ai4bharat/indictrans2-en-indic-dist-200M"
BATCH_SIZE = 128
DEVICE = "cuda"
TARGET_LANGUAGES = ["hin_Deva","tam_Taml","mar_Deva","mal_Mlym","kan_Knda"]  # Hindi, Tamil, Malayalam, Marathi, Kannada
NUM_SAMPLES = 12000
# TARGET_LANGUAGES = ["hin_Deva",]  # Hindi
# Helper Functions
def initialize_model_and_tokenizer(ckpt_dir, direction, quantization=""):
    qconfig = None
    if quantization == "4-bit":
        qconfig = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    elif quantization == "8-bit":
        qconfig = BitsAndBytesConfig(load_in_8bit=True, bnb_8bit_use_double_quant=True, bnb_8bit_compute_dtype=torch.bfloat16)

    tokenizer = IndicTransTokenizer(direction=direction)
    model = AutoModelForSeq2SeqLM.from_pretrained(ckpt_dir, trust_remote_code=True, low_cpu_mem_usage=True, quantization_config=qconfig)

    if qconfig is None:
        model = model.to(DEVICE)
        model.half()

    model.eval()
    return tokenizer, model





def batch_translate(input_sentences, src_lang, tgt_lang, model, tokenizer, initial_batch_size = BATCH_SIZE):
    translations = []
    current_batch_size = initial_batch_size
    i = 0
    pbar = tqdm(total=len(input_sentences), desc="Translating")

    while i < len(input_sentences):
        current_batch_size = initial_batch_size
        while True:
            try:
                batch = input_sentences[i: i + current_batch_size]

                # Preprocess the batch and extract entity mappings
                batch, entity_map = preprocess_batch(batch, src_lang=src_lang, tgt_lang=tgt_lang)

                # Tokenize the batch and generate input encodings
                inputs = tokenizer(batch, src=True, truncation=True, padding="longest", return_tensors="pt", return_attention_mask=True).to(DEVICE)

                # Generate translations using the model
                with torch.no_grad():
                    generated_tokens = model.generate(**inputs, use_cache=True, min_length=0, max_length=256, num_beams=5, num_return_sequences=1)

                # Decode the generated tokens into text
                generated_tokens = tokenizer.batch_decode(generated_tokens.detach().cpu().tolist(), src=False)

                # Postprocess the translations, including entity replacement
                translations += postprocess_batch(generated_tokens, lang=tgt_lang, placeholder_entity_map=entity_map)

                del inputs
                torch.cuda.empty_cache()

                pbar.update(current_batch_size)
                i += current_batch_size  # Move to the next batch
                break
            except RuntimeError as e:
                if 'out of memory' in str(e) and current_batch_size > 1:
                    logger.warning(f"OOM error caught during translation, reducing batch size to {current_batch_size // 2}")
                    current_batch_size //= 2  # Reduce the batch size
                    torch.cuda.empty_cache()
                else:
                    raise e  # If not OOM, or batch size is 1, re-raise the exception

    pbar.close()
    return translations


def gather_sentences(samples):
    sentences = []
    for sample in samples:
        for message in sample['messages']:
            if message['role'] != 'system':
                content = message['content']
                for sentence in re.split(r'(?<=[.!?;:])\s+', content):
                    if sentence:
                        sentences.append(sentence)
    return sentences

def translate_and_map(sentences, src_lang, tgt_lang, model, tokenizer):
    translations = batch_translate(sentences, src_lang, tgt_lang, model, tokenizer)
    return dict(zip(sentences, translations))


CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_FILE = "checkpoint.json"

def save_checkpoint(data, metadata, checkpoint_dir=CHECKPOINT_DIR):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    data_file = os.path.join(checkpoint_dir, f"{metadata['current_lang']}_data.pkl")
    metadata_file = os.path.join(checkpoint_dir, CHECKPOINT_FILE)

    # Save data
    pd.DataFrame(data).to_pickle(data_file)

    # Save metadata
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f)

    logger.info(f"Checkpoint saved for language {metadata['current_lang']}")

def load_checkpoint(checkpoint_dir=CHECKPOINT_DIR):
    metadata_file = os.path.join(checkpoint_dir, CHECKPOINT_FILE)
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        data_file = os.path.join(checkpoint_dir, f"{metadata['current_lang']}_data.pkl")
        data = pd.read_pickle(data_file)
        if isinstance(data, pd.DataFrame):
            data = data.to_dict(orient='records')
        return data, metadata
    return None, None


def main():
    logger.info("Loading UltraChat 200k dataset...")
    dataset = load_dataset("HuggingFaceH4/ultrachat_200k")
    
    # Select and shuffle samples
    checkpoint_data, checkpoint_metadata = load_checkpoint()
    # print("Checkpoint data sample:", checkpoint_data.tail())
    start_lang_index = 0
    if checkpoint_data is not None:
        logger.info(f"Resuming from checkpoint: {checkpoint_metadata['current_lang']}")
        selected_samples = checkpoint_data
        start_lang_index = TARGET_LANGUAGES.index(checkpoint_metadata['current_lang'])
    else:
        selected_samples = random.sample(list(dataset["train_sft"]), NUM_SAMPLES)
        random.shuffle(selected_samples)
        start_lang_index = 0
    # Split into language groups
    group_size = NUM_SAMPLES // (len(TARGET_LANGUAGES) + 1)  # +1 for English
    groups = [selected_samples[i:i + group_size] for i in range(0, NUM_SAMPLES, group_size)]

    logger.info(f"Split dataset into {len(groups)} groups of size {group_size}")

    # Initialize translation model and tokenizer
    tokenizer, model = initialize_model_and_tokenizer(en_indic_ckpt_dir, "en-indic")

    # Process for each language
    for lang_index, lang in enumerate(TARGET_LANGUAGES[start_lang_index:], start=start_lang_index):
        logger.info(f"Processing language: {lang}")
        sentences = gather_sentences(groups[lang_index])
        translations = translate_and_map(sentences, "eng_Latn", lang, model, tokenizer)
        # Replace original sentences with translated ones
        for sample in groups[TARGET_LANGUAGES.index(lang)]:
            sample['lang'] = lang
            print(sample)
            for message in sample['messages']:
                if message['role'] != 'system':
                    translated_content = ' '.join([translations.get(sentence, sentence) for sentence in re.split(r'(?<=[.!?;:])\s+', message['content'])])
                    message['content'] = translated_content
        # Save checkpoint
        checkpoint_metadata = {"current_lang": lang}
        save_checkpoint(sum(groups, []), checkpoint_metadata)

    # Combine all groups into a single dataset
    final_samples = sum(groups, [])
    final_dataset = Dataset.from_pandas(pd.DataFrame(final_samples))
    final_dataset.save_to_disk("ultrachat_multilingual")
    final_dataset.push_to_hub("rohansolo/BB-Ultrachat-IndicLingual6-12k")

if __name__ == "__main__":
    main()
