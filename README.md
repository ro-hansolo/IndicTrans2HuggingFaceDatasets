# IndicTrans2HuggingFaceDatasets
 
* currently experimenting with Dynamic batch sizes as optimum batch sizing could improve speeds drastically!


A script to use IndicTrans2 efficiently to create a balanced multilingual Indic Dataset from UltraChat 200k dataset.

 Part of an ongoing project at bhaiyabot.com to create mujltilingual Indic datasets which can improve model performance while giving ability for generalisation accross languages.

 Script was the result of struggles to efficiently use IndicTrans2 to translate samples from the UltraChat 200k dataset.

## Requirements

GPU. Or you can edit the script to use MPS instead of CUDA on Apple Silicon.

## What it does

1. Loads the UltraChat 200k dataset
2. Splits it into Equal number of samples for each language
3. Splits all text into sentences and translates them using IndicTrans2 in Batches
4. Saves the translated dataset in a HuggingFace Dataset format

## Why this script?

IndicTrans2 has a max_length of 256. Soo sentence by sentence is the only way to go. But it takes a lot of time. Soo this script uses multiprocessing to speed up the process.

This script first splits the dataset into equal number of samples for each language. Then it splits the text into sentences and translates them in batches.

Gathering all ther sentences that need to be translated and then translating them in batches is faster than translating them one by one.

The reassembly of the dataset needs to be done in a way that the order of the samples is maintained. This script does that.

Hope this helps someone, especially incase of dataset translation, hope is it can be adaped for pther datasets as well

Currently doesn't ignore code blocks. While assistant text always wraps code in backticks, user text doesn't. Soo code blocks are translated as well. This can be fixed by adding a check for backticks in the user text.

Will be changing that before the next run.

## Usage

1. Clone the repo
2. 'cd' into the repo
3. Run `source install.sh`
4. Run `python3 tanslate.py`


## Variables in the script:


# BATCH_SIZE = 256
Batch size for translation. Higher batch size means more memory usage. Lower batch size means more time taken to translate.

In case the batch_size is too high, it will be automatically reduced to the maximum possible value.

# DEVICE = "cuda"

The device to use for translation. "cuda" for GPU. "mps" for Apple Silicon.

Incase you're using mps, make sure to change `torch.cuda.clear_cache()` to `torch.mps.clear_cache()` in the script.

# TARGET_LANGUAGES = ["hin_Deva","tam_Taml","mar_Deva","mal_Mlym","kan_Knda"] 

The languages to translate to. The number of samples for each language will be equal and unique.
The sampled dataset will contain only these languages and english, in equal distribution.

# NUM_SAMPLES = 12000

Number of total samples in the dataset.

## Note

Please go through the script and change the variables as per your needs. Script was only uploaded to ensure any code produced does not go to waster or is not lost, and hopefully helps someone.

## Citations
```
@article{gala2023indictrans2,
  title   = {IndicTrans2: Towards High-Quality and Accessible Machine Translation Models for all 22 Scheduled Indian Languages},
  author  = {Jay Gala and Pranjal A. Chitale and Raghavan AK and Varun Gumma and Sumanth Doddapaneni and Aswanth Kumar and Janki Nawale and Anupama Sujatha and Ratish Puduppully and Vivek Raghavan and Pratyush Kumar and Mitesh M. Khapra and Raj Dabre and Anoop Kunchukuttan},
  year    = {2023},
  journal = {Transactions on Machine Learning Research},
  url     = {https://openreview.net/forum?id=vfT4YuzAYA}
}
```
Starting point for the script was the `IndicTrans2/huggingface_inference/example.py` repo.

```
@misc{ding2023enhancing,
      title={Enhancing Chat Language Models by Scaling High-quality Instructional Conversations}, 
      author={Ning Ding and Yulin Chen and Bokai Xu and Yujia Qin and Zhi Zheng and Shengding Hu and Zhiyuan Liu and Maosong Sun and Bowen Zhou},
      year={2023},
      eprint={2305.14233},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}

```
