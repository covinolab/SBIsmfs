#!/bin/bash
#SBATCH --job-name=SBI_test
#SBATCH --partition=test
#SBATCH --nodes=1
#SBATCH --time=00:05:00
#SBATCH --mail-type=ALL

/home/cellmembrane/dingeldein/miniconda3/envs/sbi/bin/python scr/sbi_generate_data.py --num_workers 80 --file_name test --num_sim 100
