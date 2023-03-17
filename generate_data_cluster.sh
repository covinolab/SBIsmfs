#!/bin/bash

N_worker=80
path="/scratch/cellmembrane/dingeldein/spline_potential/"
file_name="spline_"

for i in {0..15}
    do

cat << EOF > run_job.sh
#!/bin/bash
#SBATCH --job-name=SBI_splines
#SBATCH --partition=general1
#SBATCH --nodes=1
#SBATCH --time=35:00:00
#SBATCH --mail-type=ALL

/home/cellmembrane/dingeldein/miniconda3/envs/sbi/bin/python scr/sbi_generate_data.py --num_workers $N_worker --file_name "$path$file_name$i" --num_sim 100000
EOF

sbatch run_job.sh

    done

