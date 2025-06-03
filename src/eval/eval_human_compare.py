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




def average_exclude_min_max(numbers):
    if len(numbers) <= 2:
        raise ValueError("列表中的元素数量必须大于2")
    
    # 去掉最大值和最小值
    numbers_sorted = sorted(numbers)
    trimmed_numbers = numbers_sorted[1:-1]
    
    # 计算平均值并保留两位小数
    average = sum(trimmed_numbers) / len(trimmed_numbers)
    return round(average, 2)



parser = argparse.ArgumentParser()


parser.add_argument("--root_dir", default="/online1/gzs_data/Personal_file/LiZhou/TemporalCultural", type=str)
parser.add_argument("--model", default="MiniCPM-V-2_6", type=str, help=["doubao1-5-v","gpt-4o", "Qwen2.5-VL-7B-Instruct_v1", "Qwen2.5-VL-7B-Instruct","InternVL2_5-8B"])
args = parser.parse_args()

all_models = ["MiniCPM-V-2_6", "Qwen2.5-VL-7B-Instruct", "InternVL2_5-8B", "gpt-4o", "doubao1-5-v"]


# all_models = ["doubao1-5-v"]

all_instructions = {
    'svqa': ["svqa_1", "svqa_2", "svqa_3", "svqa_4", "svqa_5"],
    'mvqa': ["mvqa_1", "mvqa_2", "mvqa_3", "mvqa_4", "mvqa_5"],
}


model_results = {}
for qa, all_templates in all_instructions.items():
    model_results = {}

    human_ids_file = f"{args.root_dir}/dataset/human_evaluation/{qa}_ids.json"
    with open(human_ids_file, 'r', encoding='utf-8') as file:
        human_eval = json.load(file)

    for model in all_models:
        HUMAN_ACC = {
            'type': [],
            'gender': [],
            'period': [],
            'xiu': [],
            'jin': [],
            'ling': [],
            'bottoms': [],
            'outerwear': [],
            'overall':[]
        }

        all_results = []
        for template in all_templates:
            
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
                'type': 20,
                'gender': 20,
                'period': 20,
                'xiu': 20,
                'jin': 20,
                'ling': 20,
                'bottoms': 20,
                'outerwear': 20,
                'overall':120
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

            read_file = f"{args.root_dir}/results/{qa}/{model}/{model}_{template}.json"

            with open(read_file, 'r', encoding='utf-8') as file:
                dataset_eval = json.load(file)
            if qa == 'svqa':
                for data in tqdm(dataset_eval):
                    if data['question_id'] in human_eval:
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

                        
                        question_type = data['question_type']
                        # if question_type == 'type':
                        #     print('a')
                        # TYPE_COUNT[question_type] += 1
                        # TYPE_COUNT['overall'] += 1
                        if answer_predict[0] == data['answer']:
                            RIGHT_COUNT[question_type] += 1
                            RIGHT_COUNT['overall'] += 1

                
                # 同data
                for k, v in ACC_COUNT.items():
                    ACC_COUNT[k] = round(RIGHT_COUNT[k]/TYPE_COUNT[k]*100, 2)
                print(template)
                print(ACC_COUNT)
            else:
                for data in tqdm(dataset_eval):
                    if data['question_id'] in human_eval:
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
                        
                     
                        question_type = data['question_type']
                        # if question_type == 'type':
                        #     print('a')
                        # TYPE_COUNT[question_type] += 1
                        # TYPE_COUNT['overall'] += 1
                        
                        
                        if answer_predict[0] == ANSWER[data['answer_idx']]:
                            RIGHT_COUNT[question_type] += 1
                            RIGHT_COUNT['overall'] += 1

                
                # 同data
                for k, v in ACC_COUNT.items():
                    ACC_COUNT[k] = round(RIGHT_COUNT[k]/TYPE_COUNT[k]*100, 2)
                print(template)
                print(ACC_COUNT)

            for t, v in HUMAN_ACC.items():
                v.append(ACC_COUNT[t])
            all_results.append(ACC_COUNT['overall'])
        
        for t, v in HUMAN_ACC.items():
            HUMAN_ACC[t] = average_exclude_min_max(v)
        model_results[model]= HUMAN_ACC
        # model_results[model][qa] = average_exclude_min_max(all_results)
        # print('a')
    final_data = pd.DataFrame(model_results).T


    final_data.to_excel(f"{args.root_dir}/results/{qa}_human_compare.xlsx")

print('a')