#!/bin/bash
#SBATCH --job-name=test
#SBATCH --output=test-%j.out
#SBATCH --error=test-%j.err
#SBATCH --time=04:00:00
#SBATCH --gpus=1
#SBATCH --ntasks=64
#SBATCH --mem-per-cpu=1024
#SBATCH --mail-type=END
#SBATCH --mail-user=fscharitzer@student.ethz.ch

#load modules
module load gcc/8.2.0
module load python_gpu/3.11.2

#test SBDM
python test.py --save_model 2024_06_10_22_40_25-result_model_cno_prior_StandardNormal-min_checkpoint --dataset 'MNIST' --num_samples_mmd 1000
