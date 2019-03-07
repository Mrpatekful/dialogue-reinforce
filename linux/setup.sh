#!/bin/sh

pip install -r requirements.txt

git clone https://github.com/facebookresearch/ParlAI.git ~/ParlAI
cd ~/ParlAI; python setup.py develop
