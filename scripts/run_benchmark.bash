#!/bin/bash

# Output and metrics paths
OUTDIR=./results
METRICS=./results/metrics.csv

# Algorithms
algorithms=(
  boss_tetrad_pd2
  dagma_tetrad_pd2
  directlingam_pd2
  fges_tetrad_pd2
  grasp_tetrad_pd2
  pc_tetrad_05
)

# Generations
generations=(

#   erdos_n3_d2_rootgmm_mechbinary_noisegaussian
#   erdos_n3_d2_rootgmm_mechlinear-binary_noisegaussian
#   erdos_n3_d2_rootgmm_mechlinear_noisegaussian
#   erdos_n3_d2_rootgmm_mechnn-binary_noisegaussian
#   erdos_n3_d2_rootgmm_mechnn_noisegaussian
#   erdos_n3_d2_rootgmm_mechpolynomial-binary_noisegaussian
#   erdos_n3_d2_rootgmm_mechpolynomial_noisegaussian
#   erdos_n3_d2_rootgmm_mechsigmoid_add-binary_noisegaussian
#   erdos_n3_d2_rootgmm_mechsigmoid_add_noisegaussian
  erdos_n10_d2_rootgmm_mechbinary_noisegaussian
  erdos_n10_d2_rootgmm_mechlinear-binary_noisegaussian
  erdos_n10_d2_rootgmm_mechlinear_noisegaussian
  erdos_n10_d2_rootgmm_mechnn-binary_noisegaussian
  erdos_n10_d2_rootgmm_mechnn_noisegaussian
  erdos_n10_d2_rootgmm_mechpolynomial-binary_noisegaussian
  erdos_n10_d2_rootgmm_mechpolynomial_noisegaussian
  erdos_n10_d2_rootgmm_mechsigmoid_add-binary_noisegaussian
  erdos_n10_d2_rootgmm_mechsigmoid_add_noisegaussian
  erdos_n100_d10_rootgmm_mechbinary_noisegaussian
  erdos_n100_d10_rootgmm_mechlinear-binary_noisegaussian
  erdos_n100_d10_rootgmm_mechlinear_noisegaussian
  erdos_n100_d10_rootgmm_mechnn-binary_noisegaussian
  erdos_n100_d10_rootgmm_mechnn_noisegaussian
  erdos_n100_d10_rootgmm_mechpolynomial-binary_noisegaussian
  erdos_n100_d10_rootgmm_mechpolynomial_noisegaussian
  erdos_n100_d10_rootgmm_mechsigmoid_add-binary_noisegaussian
  erdos_n100_d10_rootgmm_mechsigmoid_add_noisegaussian
  
)

# Run all combinations
for g in "${generations[@]}"; do
  for a in "${algorithms[@]}"; do
    echo "Running generation=$g algorithm=$a"
    python3 minimal_test.py -g "$g" -a "$a" -o "$OUTDIR" -c "$METRICS"
  done
done
