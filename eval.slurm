#!/bin/bash

#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH--constraint=gpu16
#SBATCH --output=pytorch-nlp-eval-slurm-%J.out
#SBATCH --job-name=pytorch-nlp


# Output some preliminaries before we begin
date
echo "Slurm nodes: $SLURM_JOB_NODELIST"
NUM_GPUS=`echo $GPU_DEVICE_ORDINAL | tr ',' '\n' | wc -l`
echo "You were assigned $NUM_GPUS gpu(s)"

# Load the Python and CUDA modules
module load anaconda

# List the modules that are loaded
module list

# Have Nvidia tell us the GPU/CPU mapping so we know
nvidia-smi topo -m

echo

# Activate the GPU version of PyTorch
source activate nlp

chmod +x run.sh

# Run PyTorch Training
echo "Evaluation Start:"
srun run.sh
echo

# You're done!
echo "Evaluation Ends"
date

