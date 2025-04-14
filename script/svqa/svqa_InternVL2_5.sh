#!/bin/bash
#SBATCH -J svqa_InternVL
#SBATCH -p q_intel_gpu_nvidia_h20_10
#SBATCH -o svqa_InternVL.out
#SBATCH --gres=gpu:1

# module load amd/Anaconda/2023.3
cd /online1/gzs_data/Personal_file/LiZhou/TemporalCultural

conda activate hanfu

python src/svqa/open_vlm_svqa.py --model_name InternVL2_5-8B --instruction svqa_1 svqa_2 svqa_3 svqa_cot svqa_rationale svqa_en 