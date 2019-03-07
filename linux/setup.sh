#!/bin/sh

# Installs the required libraries, including ParlAI.

pip install -r $(dirname "$0")/../requirements.txt

pip install torch torchvision

git clone https://github.com/facebookresearch/ParlAI.git ~/ParlAI
python ~/ParlAI/setup.py develop
