#!/bin/bash
#SBATCH --job-name=causal_discovery_benchmark
#SBATCH --output=slurm_logs/job_%A_%a.out
#SBATCH --error=slurm_logs/job_%A_%a.err
#SBATCH --array=1-216
#SBATCH --time=24:00:00

# Create logs directory if it doesn't exist
mkdir -p slurm_logs

# Create results directory if it doesn't exist
mkdir -p results

# Define all generation configurations
GENERATIONS=(
    "erdos_n5_d2_rootgmm_mechbinary_noisegaussian"
    "erdos_n5_d2_rootgmm_mechlinear-binary_noisegaussian"
    "erdos_n5_d2_rootgmm_mechlinear_noisegaussian"
    "erdos_n5_d2_rootgmm_mechnn-binary_noisegaussian"
    "erdos_n5_d2_rootgmm_mechnn_noisegaussian"
    "erdos_n5_d2_rootgmm_mechpolynomial-binary_noisegaussian"
    "erdos_n5_d2_rootgmm_mechpolynomial_noisegaussian"
    "erdos_n5_d2_rootgmm_mechsigmoid_add-binary_noisegaussian"
    "erdos_n5_d2_rootgmm_mechsigmoid_add_noisegaussian"
    "erdos_n10_d2_rootgmm_mechbinary_noisegaussian"
    "erdos_n10_d2_rootgmm_mechlinear-binary_noisegaussian"
    "erdos_n10_d2_rootgmm_mechlinear_noisegaussian"
    "erdos_n10_d2_rootgmm_mechnn-binary_noisegaussian"
    "erdos_n10_d2_rootgmm_mechnn_noisegaussian"
    "erdos_n10_d2_rootgmm_mechpolynomial-binary_noisegaussian"
    "erdos_n10_d2_rootgmm_mechpolynomial_noisegaussian"
    "erdos_n10_d2_rootgmm_mechsigmoid_add-binary_noisegaussian"
    "erdos_n10_d2_rootgmm_mechsigmoid_add_noisegaussian"
    "erdos_n20_d2_rootgmm_mechbinary_noisegaussian"
    "erdos_n20_d2_rootgmm_mechlinear-binary_noisegaussian"
    "erdos_n20_d2_rootgmm_mechlinear_noisegaussian"
    "erdos_n20_d2_rootgmm_mechnn-binary_noisegaussian"
    "erdos_n20_d2_rootgmm_mechnn_noisegaussian"
    "erdos_n20_d2_rootgmm_mechpolynomial-binary_noisegaussian"
    "erdos_n20_d2_rootgmm_mechpolynomial_noisegaussian"
    "erdos_n20_d2_rootgmm_mechsigmoid_add-binary_noisegaussian"
    "erdos_n20_d2_rootgmm_mechsigmoid_add_noisegaussian"
    "erdos_n40_d2_rootgmm_mechbinary_noisegaussian"
    "erdos_n40_d2_rootgmm_mechlinear-binary_noisegaussian"
    "erdos_n40_d2_rootgmm_mechlinear_noisegaussian"
    "erdos_n40_d2_rootgmm_mechnn-binary_noisegaussian"
    "erdos_n40_d2_rootgmm_mechnn_noisegaussian"
    "erdos_n40_d2_rootgmm_mechpolynomial-binary_noisegaussian"
    "erdos_n40_d2_rootgmm_mechpolynomial_noisegaussian"
    "erdos_n40_d2_rootgmm_mechsigmoid_add-binary_noisegaussian"
    "erdos_n40_d2_rootgmm_mechsigmoid_add_noisegaussian"
    "erdos_n50_d2_rootgmm_mechbinary_noisegaussian"
    "erdos_n50_d2_rootgmm_mechlinear-binary_noisegaussian"
    "erdos_n50_d2_rootgmm_mechlinear_noisegaussian"
    "erdos_n50_d2_rootgmm_mechnn-binary_noisegaussian"
    "erdos_n50_d2_rootgmm_mechnn_noisegaussian"
    "erdos_n50_d2_rootgmm_mechpolynomial-binary_noisegaussian"
    "erdos_n50_d2_rootgmm_mechpolynomial_noisegaussian"
    "erdos_n50_d2_rootgmm_mechsigmoid_add-binary_noisegaussian"
    "erdos_n50_d2_rootgmm_mechsigmoid_add_noisegaussian"
)

# Define all algorithm configurations
ALGORITHMS=(
    "boss_tetrad_pd2"
    "dagma_tetrad_pd2"
    "directlingam_pd2"
    "fges_tetrad_pd2"
    "grasp_tetrad_pd2"
    "pc_tetrad_05"
)

# Calculate which generation and algorithm to use for this array task
# Array tasks are 1-indexed, so subtract 1 to get 0-indexed
TASK_IDX=$((SLURM_ARRAY_TASK_ID - 1))

# Calculate generation and algorithm indices
# Each generation gets tested with all algorithms
GENERATION_IDX=$((TASK_IDX / ${#ALGORITHMS[@]}))
ALGORITHM_IDX=$((TASK_IDX % ${#ALGORITHMS[@]}))

# Get the specific generation and algorithm names
GENERATION_NAME="${GENERATIONS[$GENERATION_IDX]}"
ALGORITHM_NAME="${ALGORITHMS[$ALGORITHM_IDX]}"

# Print job information
echo "Starting job array task $SLURM_ARRAY_TASK_ID of $SLURM_ARRAY_TASK_MAX"
echo "Job ID: $SLURM_ARRAY_JOB_ID"
echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Running on node: $SLURMD_NODENAME"
echo "Working directory: $PWD"
echo "Date: $(date)"
echo ""
echo "Task breakdown:"
echo "  Generation index: $GENERATION_IDX"
echo "  Algorithm index: $ALGORITHM_IDX"
echo "  Generation: $GENERATION_NAME"
echo "  Algorithm: $ALGORITHM_NAME"
echo ""

# Create output directory for this specific experiment
OUTPUT_DIR="results/${GENERATION_NAME}/${ALGORITHM_NAME}"
mkdir -p "$OUTPUT_DIR"

# Create CSV file for results
CSV_FILE="results/benchmark_results.csv"

# Construct the command
COMMAND="python benchmark_synthetic.py --generation $GENERATION_NAME --algorithm $ALGORITHM_NAME --output-dir $OUTPUT_DIR --csv $CSV_FILE --retries 3"

echo "Executing command: $COMMAND"
echo "----------------------------------------"

# Execute the command
eval $COMMAND

# Check exit status
if [ $? -eq 0 ]; then
    echo "Command completed successfully"
    echo "Results saved to: $OUTPUT_DIR"
    echo "CSV results appended to: $CSV_FILE"
else
    echo "Command failed with exit code $?"
    exit 1
fi

echo "Job completed at: $(date)"