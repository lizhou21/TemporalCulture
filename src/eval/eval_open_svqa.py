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
parser.add_argument("--model", default="Qwen2.5-VL-7B-Instruct", type=str, help=["doubao1-5-v","gpt-4o", "Qwen2.5-VL-7B-Instruct", "Qwen2.5-VL-7B-Instruct","InternVL2_5-8B"])
args = parser.parse_args()

# all_instructions = ["svqa_1", "svqa_2", "svqa_3", "svqa_4", "svqa_5", "svqa_cot", "svqa_cot1","svqa_rationale", "svqa_en"]
all_instructions = ["svqa_1", "svqa_2", "svqa_3", "svqa_4", "svqa_5", "svqa_cot", "svqa_cot1","svqa_rationale", "svqa_en", "svqa_with_face", "svqa_face_rationale", "svqa_extra_info"]

# all_instructions = ["svqa_1", "svqa_2", "svqa_3", "svqa_cot", "svqa_rationale", "svqa_en"]
# all_instructions = ["svqa_1", "svqa_2", "svqa_3"]

all_results = {}
for template in all_instructions:
    ACC_COUNT = {
        'type': 0,
        'gender': 0,
        'period': 0,
        'xiu': 0,
        'jin': 0,
        'ling': 0,
        'bottoms': 0,
        'outerwear': 0,
        'overall':0
    }
    TYPE_COUNT = {
        'type': 0,
        'gender': 0,
        'period': 0,
        'xiu': 0,
        'jin': 0,
        'ling': 0,
        'bottoms': 0,
        'outerwear': 0,
        'overall':0
    }

    RIGHT_COUNT = {
        'type': 0,
        'gender': 0,
        'period': 0,
        'xiu': 0,
        'jin': 0,
        'ling': 0,
        'bottoms': 0,
        'outerwear': 0,
        'overall':0
    }

    read_file = f"{args.root_dir}/results/svqa/{args.model}/{args.model}_{template}.json"

    with open(read_file, 'r', encoding='utf-8') as file:
        dataset_eval = json.load(file)
    for data in tqdm(dataset_eval):
        if "en" not in template:
            if args.model == 'gpt-4o':
                predict = data['predict'][7:-3]
                answer_predict = json.loads(predict)['答案']

            elif args.model == 'doubao1-5-v':
                predict = data['predict']
                answer_predict = json.loads(predict)['答案']
            elif args.model == 'Qwen2.5-VL-7B-Instruct_v1':
                predict = data['predict'][0]+"\"\n}\n"
                predict = predict[7:]
                answer_predict = json.loads(predict)['答案']
            elif args.model == "InternVL2_5-8B":
                predict = data['predict'][7:-3]
                answer_predict = json.loads(predict)['答案']
            elif args.model == "Qwen2.5-VL-7B-Instruct":
                predict = data['predict'][0][7:-3]
                answer_predict = json.loads(predict)['答案']
            elif args.model == "Llama-3.2-11B-Vision":

                # predict = re.split(r'\n+',data['predict'].strip())
                if type(data['predict'])==list:
                    predict = re.split(r'##',data['predict'][0].strip())
                else:
                    predict = re.split(r'##',data['predict'].strip())
                answer_predict = predict[2].split("答案：")[-1].strip()
                # answer_predict = predict[2].split("答案：")[1].strip()
            elif args.model in ["MiniCPM-Llama3-V-2_5", "MiniCPM-V-2_6"]:
                ans = []
                if "A" in data['predict']:
                    ans.append("A")
                elif "B" in data['predict']:
                    ans.append("B")
                elif "C" in data['predict']:
                    ans.append("C")
                elif "D" in data['predict']:
                    ans.append("D")
                
                if len(ans) == 1:
                    answer_predict = ans[0]
                elif len(ans) == 0:
                    answer_predict == "0"
                else:
                    print('a')
                # predict = data['predict'][0][7:-3]
                # answer_predict = json.loads(predict)['答案']

        
        else:
            if args.model == 'gpt-4o':
                predict = data['predict'][7:-3]
                answer_predict = json.loads(predict)['answer']

            elif args.model == 'doubao1-5-v':
                predict = data['predict']
                answer_predict = json.loads(predict)['answer']
            elif args.model == 'Qwen2.5-VL-7B-Instruct_v1':
                predict = data['predict'][0]+"\"\n}\n"
                predict = predict[7:]
                answer_predict = json.loads(predict)['answer']
            elif args.model == "InternVL2_5-8B":
                predict = data['predict'][7:-3]
                answer_predict = json.loads(predict)['answer']
            elif args.model == "Qwen2.5-VL-7B-Instruct":
                predict = data['predict'][0][7:-3]
                answer_predict = json.loads(predict)['answer']
            elif args.model in ["MiniCPM-Llama3-V-2_5", "MiniCPM-V-2_6"]:
                ans = []
                predict = data['predict'].split("\n")[-1]
                if "A" in predict:
                    ans.append("A")
                elif "B" in predict:
                    ans.append("B")
                elif "C" in predict:
                    ans.append("C")
                elif "D" in predict:
                    ans.append("D")
                
                if len(ans) == 1:
                    answer_predict = ans[0]
                elif len(ans) == 0:
                    answer_predict = "0"
                else:
                    print('a')
            

        question_type = data['question_type']
        # if question_type == 'type':
        #     print('a')
        TYPE_COUNT[question_type] += 1
        TYPE_COUNT['overall'] += 1
        if answer_predict[0] == data['answer']:
            RIGHT_COUNT[question_type] += 1
            RIGHT_COUNT['overall'] += 1

    for k, v in ACC_COUNT.items():
        ACC_COUNT[k] = round(RIGHT_COUNT[k]/TYPE_COUNT[k]*100, 2)
    print(template)
    print(ACC_COUNT)
    all_results[template] = ACC_COUNT
final_data = pd.DataFrame(all_results).T


final_data.to_excel(f"{args.root_dir}/results/svqa/{args.model}/{args.model}.xlsx")

print('a')