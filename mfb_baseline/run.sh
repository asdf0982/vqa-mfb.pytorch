#!/bin/bash

LOG=log/mfb_baseline-`date +%Y-%m-%d-%H-%M-%S`.log
python train_mfb_baseline.py \
        2>&1 | tee $LOG	
