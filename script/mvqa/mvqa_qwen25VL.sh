#!/bin/bash
#SBATCH -J mvqa_qwen
#SBATCH -p q_intel_gpu_nvidia_h20_10
#SBATCH -o mvqa_qwen.out
#SBATCH --gres=gpu:1
#SBATCH --mem=45G

source activate
# 退出虚拟环境
conda deactivate
conda init
conda activate hanfu
# module load amd/Anaconda/2023.3
cd /online1/gzs_data/Personal_file/LiZhou/TemporalCultural


python src/mvqa/open_vlm_mvqa.py --model_name Qwen2.5-VL-7B-Instruct --instruction mvqa_1 mvqa_2