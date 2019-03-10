#!/bin/sh

MODEL=${1:-seq2seq}
TASK=${2:-dailydialog}

MODEL_DIR=$(dirname "$0")/../checkpoints/$MODEL
MODEL_FILE=${3:-$MODEL_DIR/"model"}

python $(dirname "$0")/../rlchat/train.py --model $MODEL \
                                          --task $TASK \
                                          --model_file $MODEL_FILE
