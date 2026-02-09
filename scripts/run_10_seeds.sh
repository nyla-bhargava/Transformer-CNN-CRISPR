#!/bin/bash
# Train 10 models for the proposed Stage-1 Gated method
for seed in {0..9}
do
   python train.py --seed $seed --use_stage1
done

# Train 10 models for the Baseline
for seed in {0..9}
do
   python train.py --seed $seed
done
