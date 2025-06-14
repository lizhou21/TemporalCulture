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



parser = argparse.ArgumentParser()


parser.add_argument("--root_dir", default="/data1/home/lizhou/project/TemporalCultural", type=str)
parser.add_argument("--model", default="doubao1-5-v", type=str, help=["doubao1-5-v","gpt-4o", "Qwen2.5-VL-7B-Instruct_v1", "Qwen2.5-VL-7B-Instruct","InternVL2_5-8B"])
args = parser.parse_args()

# all_instructions = ["svqa_1", "svqa_2", "svqa_3","svqa_cot", "svqa_rationale", "svqa_en"]
all_instructions = ["svqa_1", "svqa_2", "svqa_3", "svqa_4", "svqa_5", "svqa_cot", "svqa_cot1", "svqa_rationale", "svqa_en", "svqa_with_face", "svqa_face_rationale", "before"]
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