#!/bin/bash
#SBATCH --job-name=train
#SBATCH --output=train-%j.out
#SBATCH --error=train-%j.err
#SBATCH --time=120:00:00
#SBATCH --gpus=1
#SBATCH --ntasks=64
#SBATCH --mem-per-cpu=1024
#SBATCH --mail-type=END
#SBATCH --mail-user=fscharitzer@student.ethz.ch

#load modules
module load gcc/8.2.0
module load python_gpu/3.11.2

#train SBDM
python main.py --dataset 'MNIST' --lr 0.0001 --batch_size 256 --prior_name 'standard' --model 'fno' --modes 8 --seed 1 --num_samples_mmd 1000
