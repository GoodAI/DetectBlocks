#!/bin/bash
#SBATCH --partition=RMC-C01-BATCH 
#SBATCH --output=/net/rmc-lx0318/home_local/tkac_ka/slurm_out/nc-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60000M
#SBATCH --gres=gpu:1
#SBATCH --time=200:00:00
#SBATCH --exclude=rmc-gpu01,rmc-gpu07,rmc-gpu13,rmc-gpu15,rmc-gpu03

python /net/rmc-lx0318/home_local/tkac_ka/siamese/master-thesis/contrastive.py --exp_name nor --lr 1e-6 -simple_network --epochs 5000
