#/bin/bash

root_dir=$(pwd)
echo "Setting up the environment in the $root_dir"

# --------------------------------------------------------------
#          create and activate the virtual environment
# --------------------------------------------------------------
# Create a virtual environment with python3
echo "Creating a virtual environment with python3"
python3 -m venv bbaiprojdataenv

# Activate the virtual environment
echo "Activating the virtual environment"
source bbaiprojdataenv/bin/activate

# Upgrade pip in the virtual environment
echo "Upgrading pip"
python3 -m pip install --upgrade pip


# --------------------------------------------------------------
#       Install IndicNLP library and necessary resources
# --------------------------------------------------------------
git clone https://github.com/anoopkunchukuttan/indic_nlp_resources.git
export INDIC_RESOURCES_PATH=$root_dir/indic_nlp_resources

# we use version 0.92 which is the latest in the github repo
git clone https://github.com/anoopkunchukuttan/indic_nlp_library.git
cd indic_nlp_library
python3 -m pip install ./
cd $root_dir


# --------------------------------------------------------------
#               Install additional utility packages
# --------------------------------------------------------------
python3 -m pip install nltk sacremoses pandas regex mock transformers==4.33.2 urduhack[tf] mosestokenizer
python3 -c "import urduhack; urduhack.download()"
python3 -c "import nltk; nltk.download('punkt')"
python3 -m pip install bitsandbytes scipy accelerate datasets


# --------------------------------------------------------------
#               Sentencepiece for tokenization
# --------------------------------------------------------------
# build the cpp binaries from the source repo in order to use the command line utility
# source repo: https://github.com/google/sentencepiece
python3 -m pip install sentencepiece
git submodule add https://github.com/AI4Bharat/IndicTrans2.git


echo "Setup completed!"
