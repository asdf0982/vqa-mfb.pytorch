#!/bin/bash

LOG=log/mfh_baseline-`date +%Y-%m-%d-%H-%M-%S`.log
python train_mfh_baseline.py \
        2>&1 | tee $LOG	
