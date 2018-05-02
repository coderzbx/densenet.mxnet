#!/bin/bash

prefix="/data/deeplearning/dataset/arrow/train_0307"

python preprocess/trans_arrow_data_2.py "${prefix}" "/data/deeplearning/dataset/arrow/train" \
	--num-thread=4 \
	--center-pad

