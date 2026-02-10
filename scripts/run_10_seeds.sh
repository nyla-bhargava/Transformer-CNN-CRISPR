%%bash

# Array of seeds for statistical significance
SEEDS=(0 1 2 3 4 5 6 7 8 9)

for s in "${SEEDS[@]}"
do
  echo "------------------------------------------"
  echo "PROCESSING SEED: $s"
  echo "------------------------------------------"

  # 1. Train Baseline
  echo "Training BASELINE..."
  PYTHONPATH=. python stage2/train.py --seed $s

  # 2. Evaluate Baseline
  echo "Evaluating BASELINE..."
  PYTHONPATH=. python stage2/evaluate.py --seed $s

  # 3. Train Stage-1 (The Improvement)
  echo "Training STAGE-1 ENABLED..."
  PYTHONPATH=. python stage2/train.py --seed $s --use_stage1

  # 4. Evaluate Stage-1
  echo "Evaluating STAGE-1..."
  PYTHONPATH=. python stage2/evaluate.py --seed $s --use_stage1
done

echo "All experiments complete. Check the 'results/' folder."
