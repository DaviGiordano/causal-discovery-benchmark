#!/bin/bash
#SBATCH --job-name=causal_discovery_benchmark_continuous
#SBATCH --output=slurm_logs_continuous/job_%A_%a.out
#SBATCH --error=slurm_logs_continuous/job_%A_%a.err
#SBATCH --array=1-180
#SBATCH --time=24:00:00

mkdir -p slurm_logs_continuous
mkdir -p results_tiers

GENERATIONS=(
#   "erdos_n5_d2_rootgmm_mechbinary_noisegaussian"
#   "erdos_n5_d2_rootgmm_mechlinear-binary_noisegaussian"
#   "erdos_n5_d2_rootgmm_mechlinear_noisegaussian"
#   "erdos_n5_d2_rootgmm_mechnn-binary_noisegaussian"
#   "erdos_n5_d2_rootgmm_mechnn_noisegaussian"
#   "erdos_n5_d2_rootgmm_mechpolynomial-binary_noisegaussian"
#   "erdos_n5_d2_rootgmm_mechpolynomial_noisegaussian"
#   "erdos_n5_d2_rootgmm_mechsigmoid_add-binary_noisegaussian"
#   "erdos_n5_d2_rootgmm_mechsigmoid_add_noisegaussian"
#   "erdos_n10_d2_rootgmm_mechbinary_noisegaussian"
#   "erdos_n10_d2_rootgmm_mechlinear-binary_noisegaussian"
#   "erdos_n10_d2_rootgmm_mechlinear_noisegaussian"
#   "erdos_n10_d2_rootgmm_mechnn-binary_noisegaussian"
#   "erdos_n10_d2_rootgmm_mechnn_noisegaussian"
#   "erdos_n10_d2_rootgmm_mechpolynomial-binary_noisegaussian"
#   "erdos_n10_d2_rootgmm_mechpolynomial_noisegaussian"
#   "erdos_n10_d2_rootgmm_mechsigmoid_add-binary_noisegaussian"
#   "erdos_n10_d2_rootgmm_mechsigmoid_add_noisegaussian"
#   "erdos_n20_d2_rootgmm_mechbinary_noisegaussian"
#   "erdos_n20_d2_rootgmm_mechlinear-binary_noisegaussian"
#   "erdos_n20_d2_rootgmm_mechlinear_noisegaussian"
#   "erdos_n20_d2_rootgmm_mechnn-binary_noisegaussian"
#   "erdos_n20_d2_rootgmm_mechnn_noisegaussian"
#   "erdos_n20_d2_rootgmm_mechpolynomial-binary_noisegaussian"
#   "erdos_n20_d2_rootgmm_mechpolynomial_noisegaussian"
#   "erdos_n20_d2_rootgmm_mechsigmoid_add-binary_noisegaussian"
#   "erdos_n20_d2_rootgmm_mechsigmoid_add_noisegaussian"
  "erdos_n30_d2_rootgmm_mechbinary_noisegaussian"
  "erdos_n30_d2_rootgmm_mechlinear-binary_noisegaussian"
  "erdos_n30_d2_rootgmm_mechlinear_noisegaussian"
  "erdos_n30_d2_rootgmm_mechnn-binary_noisegaussian"
  "erdos_n30_d2_rootgmm_mechnn_noisegaussian"
  "erdos_n30_d2_rootgmm_mechpolynomial-binary_noisegaussian"
  "erdos_n30_d2_rootgmm_mechpolynomial_noisegaussian"
  "erdos_n30_d2_rootgmm_mechsigmoid_add-binary_noisegaussian"
  "erdos_n30_d2_rootgmm_mechsigmoid_add_noisegaussian"
#   "erdos_n40_d2_rootgmm_mechbinary_noisegaussian"
#   "erdos_n40_d2_rootgmm_mechlinear-binary_noisegaussian"
#   "erdos_n40_d2_rootgmm_mechlinear_noisegaussian"
#   "erdos_n40_d2_rootgmm_mechnn-binary_noisegaussian"
#   "erdos_n40_d2_rootgmm_mechnn_noisegaussian"
#   "erdos_n40_d2_rootgmm_mechpolynomial-binary_noisegaussian"
#   "erdos_n40_d2_rootgmm_mechpolynomial_noisegaussian"
#   "erdos_n40_d2_rootgmm_mechsigmoid_add-binary_noisegaussian"
#   "erdos_n40_d2_rootgmm_mechsigmoid_add_noisegaussian"
#   "erdos_n50_d2_rootgmm_mechbinary_noisegaussian"
#   "erdos_n50_d2_rootgmm_mechlinear-binary_noisegaussian"
#   "erdos_n50_d2_rootgmm_mechlinear_noisegaussian"
#   "erdos_n50_d2_rootgmm_mechnn-binary_noisegaussian"
#   "erdos_n50_d2_rootgmm_mechnn_noisegaussian"
#   "erdos_n50_d2_rootgmm_mechpolynomial-binary_noisegaussian"
#   "erdos_n50_d2_rootgmm_mechpolynomial_noisegaussian"
#   "erdos_n50_d2_rootgmm_mechsigmoid_add-binary_noisegaussian"
#   "erdos_n50_d2_rootgmm_mechsigmoid_add_noisegaussian"
)

ALGORITHMS=(
  "boss_tetrad_pd2"
  "fges_tetrad_pd2"
  "grasp_tetrad_pd2"
  "pc_tetrad_05"
)

NUM_TIERS=("1" "2" "3" "4" "5")

TASK_IDX=$((SLURM_ARRAY_TASK_ID - 1))

num_gen=${#GENERATIONS[@]}
num_alg=${#ALGORITHMS[@]}
num_tiers=${#NUM_TIERS[@]}

GENERATION_IDX=$(( TASK_IDX / (num_alg * num_tiers) ))
ALGORITHM_IDX=$(( (TASK_IDX / num_tiers) % num_alg ))
TIER_IDX=$(( TASK_IDX % num_tiers ))

GENERATION_NAME="${GENERATIONS[$GENERATION_IDX]}"
ALGORITHM_NAME="${ALGORITHMS[$ALGORITHM_IDX]}"
TIER="${NUM_TIERS[$TIER_IDX]}"

echo "Task $SLURM_ARRAY_TASK_ID | gen=$GENERATION_NAME alg=$ALGORITHM_NAME tiers=$TIER"

OUTPUT_DIR="results_tiers/${GENERATION_NAME}/${ALGORITHM_NAME}/t${TIER}"
mkdir -p "$OUTPUT_DIR"

CSV_FILE="results_tiers/benchmark_results.csv"

COMMAND="python benchmark_synthetic_continuous.py \
  --generation $GENERATION_NAME \
  --algorithm $ALGORITHM_NAME \
  --output-dir $OUTPUT_DIR \
  --csv $CSV_FILE \
  -t $TIER \
  --retries 3"

echo "Executing: $COMMAND"
eval $COMMAND
status=$?

if [ $status -eq 0 ]; then
  echo "OK -> $OUTPUT_DIR (CSV: $CSV_FILE)"
else
  echo "Failed with exit code $status"
  exit $status
fi

echo "Done at: $(date)"
