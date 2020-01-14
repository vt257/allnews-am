#!/bin/bash

if [ "$1" = "train" ]; then
	CUDA_VISIBLE_DEVICES=0 python run.py train --train-src=pioner-silver/train.conll03  \
         --cuda --full-finetuning
elif [ "$1" = "test" ]; then
    CUDA_VISIBLE_DEVICES=0 python run.py evaluate model.bin pioner-silver/dev.conll03 --cuda
else
	echo "Invalid Option Selected"
fi