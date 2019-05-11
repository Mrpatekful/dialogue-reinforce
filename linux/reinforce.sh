#!/bin/sh

MODEL=${1:-transformer/generator}
TASK=${2:-dailydialog:no_start}

MODEL_DIR=$(dirname "$0")/../checkpoints/transformer
MODEL_FILE=${3:-$MODEL_DIR/"model"}

python $(dirname "$0")/../rlchat/train.py --model $MODEL \
                                          --task $TASK \
                                          --model_file $MODEL_FILE
