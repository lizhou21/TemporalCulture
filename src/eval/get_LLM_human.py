from openai import OpenAI
import pandas as pd
import json
from tqdm import tqdm
import argparse

import requests
import base64

import re
import os
import io









parser = argparse.ArgumentParser()


parser.add_argument("--root_dir", default="/online1/gzs_data/Personal_file/LiZhou/TemporalCultural", type=str)
parser.add_argument("--model", default="MiniCPM-V-2_6", type=str, help=["doubao1-5-v","gpt-4o", "Qwen2.5-VL-7B-Instruct_v1", "Qwen2.5-VL-7B-Instruct","InternVL2_5-8B"])
args = parser.parse_args()

all_models = ["MiniCPM-V-2_6", "Qwen2.5-VL-7B-Instruct", "InternVL2_5-8B", "gpt-4o", "doubao1-5-v"]




all_instructions = {
    'svqa': ["svqa_1", "svqa_2", "svqa_3", "svqa_4", "svqa_5"],
    'mvqa': ["mvqa_1", "mvqa_2", "mvqa_3", "mvqa_4", "mvqa_5"],
}
save_dir = "/online1/gzs_data/Personal_file/LiZhou/TemporalCultural/dataset/human_evaluation"

model_results = {}
for qa, all_templates in all_instructions.items():
    model_results = {}

    human_ids_file = f"{args.root_dir}/dataset/human_evaluation/{qa}_ids.json"
    with open(human_ids_file, 'r', encoding='utf-8') as file:
        human_eval = json.load(file)

    for model in all_models:
        new_save_model_dir = save_dir + '/' + model
        if not os.path.exists(new_save_model_dir):
            # 创建文件夹
            os.makedirs(new_save_model_dir)

        
        all_results = []
        for template in all_templates:
            save_data = []
            
            read_file = f"{args.root_dir}/results/{qa}/{model}/{model}_{template}.json"

            with open(read_file, 'r', encoding='utf-8') as file:
                dataset_eval = json.load(file)
            if qa == 'svqa':
                for data in tqdm(dataset_eval):
                    if data['question_id'] in human_eval:
                        save_data.append(data)
                
                
                
            else:
                for data in tqdm(dataset_eval):
                    if data['question_id'] in human_eval:
                        save_data.append(data)
            with open(f"{new_save_model_dir}/{qa}_{model}_{template}.json", 'w', encoding='utf-8') as f:
                    json.dump(save_data, f, ensure_ascii=False, indent=2)

        #     merged_json_filename = "merged_sivqa.json"
        # with open(merged_json_filename, 'w', encoding='utf-8') as f:
        #     json.dump(all_questions, f, ensure_ascii=False, indent=2)
                        
    # final_data.to_excel(f"{args.root_dir}/results/{qa}_human_compare.xlsx")

print('a')