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


ANSWER = {
    0: "A",
    1: 'B',
    2: "C",
    3: "D"
}




parser = argparse.ArgumentParser()


parser.add_argument("--root_dir", default="/online1/gzs_data/Personal_file/LiZhou/TemporalCultural", type=str)
parser.add_argument("--model", default="MiniCPM-V-2_6", type=str, help=["doubao1-5-v","gpt-4o", "Qwen2.5-VL-7B-Instruct_v1", "Qwen2.5-VL-7B-Instruct","InternVL2_5-8B"])
args = parser.parse_args()

# all_instructions = ["svqa_1", "svqa_2", "svqa_3","svqa_cot", "svqa_rationale", "svqa_en"]
all_instructions = ["mvqa_1", "mvqa_2", "mvqa_3", "mvqa_4", "mvqa_5", "mvqa_cot", "mvqa_rationale", "mvqa_en", "mvqa_with_face", "mvqa_face_rationale"]
# all_instructions = ["mvqa_cot", "mvqa_rationale", "mvqa_face_rationale"]

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

    read_file = f"{args.root_dir}/results/mvqa/{args.model}/{args.model}_{template}.json"

    with open(read_file, 'r', encoding='utf-8') as file:
        dataset_eval = json.load(file)
    for data in tqdm(dataset_eval):
        if "en" not in template:
            if args.model == 'gpt-4o':
                predict = data['predict'][7:-3]
                answer_predict = json.loads(predict)['答案']

            elif args.model == 'doubao1-5-v':
                if data['predict'][0] == "{":
                    predict = data['predict']
                    answer_predict = json.loads(predict)['答案']
                else:
                    continue
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
            elif args.model == "gemini-25-p":
                if data['predict'].startswith("<think>"):
                    predict = data['predict'].split('```json')[-1][0:-3]
                    answer_predict = json.loads(predict)['答案']
                else:
                    predict = data['predict'][7:-3]
                    answer_predict = json.loads(predict)['答案']
            elif args.model == "deepseek-r1":
                if data['predict'].startswith("<think>"):
                    predict = data['predict'].split('```json')[-1][0:-3]
                    answer_predict = json.loads(predict)['答案']
                else:
                    predict = data['predict'][7:-3]
                    answer_predict = json.loads(predict)['答案']
            elif args.model in ["MiniCPM-Llama3-V-2_5", "MiniCPM-V-2_6"]:
                if template not in ["mvqa_cot", "mvqa_rationale", "mvqa_face_rationale"]:
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
                else:
                    ans = []
                    predict = data['predict'].split("\n")[-2]
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
            elif args.model == "gemini-25-p":
                predict = data['predict'][7:-3]
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
        
        
        if answer_predict[0] == ANSWER[data['answer_idx']]:
            RIGHT_COUNT[question_type] += 1
            RIGHT_COUNT['overall'] += 1

    for k, v in ACC_COUNT.items():
        ACC_COUNT[k] = round(RIGHT_COUNT[k]/TYPE_COUNT[k]*100, 2)
    print(template)
    print(ACC_COUNT)
    all_results[template] = ACC_COUNT
final_data = pd.DataFrame(all_results).T


final_data.to_excel(f"{args.root_dir}/results/mvqa/{args.model}/{args.model}.xlsx")

print('a')