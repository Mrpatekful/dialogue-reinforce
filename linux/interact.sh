#!/bin/bash

MODEL=${1:-seq2seq}
TASK=${2:-dailydialog}

MODEL_FILE=${3:-$MODEL_DIR/$MODEL}

python ~/ParlAI/parlai/scripts/train_model.py --model $MODEL \
                                              --task $TASK \
                                              --model_file $MODEL_FILE
