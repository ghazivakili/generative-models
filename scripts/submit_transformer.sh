#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=4000
#SBATCH --job-name=drug_ch42
#SBATCH --output=logs/drug_ch42_%j.txt
#SBATCH --error=logs/drug_ch42_%j.err

module load gcccore gompi/2022a python/3.10.4 pytorch syba
python qcbm_lstm_version_sara.py qcbm 10
