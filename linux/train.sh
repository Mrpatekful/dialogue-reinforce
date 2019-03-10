#!/bin/bash

MODEL=${1:-seq2seq}
TASK=${2:-dailydialog}

MODEL_DIR=$(dirname "$0")/../checkpoints/$MODEL
MODEL_FILE=${3:-$MODEL_DIR/"model"}

mkdir -p $(dirname "$0")/../checkpoints
mkdir -p $MODEL_DIR

python ~/ParlAI/parlai/scripts/train_model.py --model $MODEL \
                                              --task $TASK \
                                              --model_file $MODEL_FILE
