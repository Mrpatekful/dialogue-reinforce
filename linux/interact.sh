#!/bin/bash

MODEL=${1:-transformer/generator}
TASK=${2:-dailydialog:no_start}

MODEL_DIR=$(dirname "$0")/../checkpoints/transformer
MODEL_FILE=${3:-$MODEL_DIR/"model"}

mkdir -p $(dirname "$0")/../checkpoints
mkdir -p $MODEL_DIR

python ~/ParlAI/parlai/scripts/interactive.py --model $MODEL \
                                              --task $TASK \
                                              --model_file $MODEL_FILE
