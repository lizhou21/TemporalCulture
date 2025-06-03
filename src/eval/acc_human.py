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





all_results = {
    "svqa":{},
    "mvqa":{}
}
folder_path = "/online1/gzs_data/Personal_file/LiZhou/TemporalCultural/dataset/human_evaluation/evaluators"

file_names = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

for f in file_names:
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
    file_na = f"{folder_path}/{f}"
    with open(file_na, 'r', encoding='utf-8') as file:
        dataset_eval = json.load(file)
    for d in tqdm(dataset_eval):
        question_type = d['question_type']
        TYPE_COUNT[question_type] += 1
        TYPE_COUNT['overall'] += 1
        if "sivqa" in f:
            human_answer = d['human_answer'].strip()
            if  human_answer== d['answer']:
                RIGHT_COUNT[question_type] += 1
                RIGHT_COUNT['overall'] += 1
            else:
                print('a')
        else:
            if d['human_answer'] == d['answer_idx']:
                RIGHT_COUNT[question_type] += 1
                RIGHT_COUNT['overall'] += 1
            else:
                print('a')
    for k, v in ACC_COUNT.items():
        ACC_COUNT[k] = round(RIGHT_COUNT[k]/TYPE_COUNT[k]*100, 2)
    if "sivqa" in f:
        all_results['svqa'][f] = ACC_COUNT
    else:
        all_results['mvqa'][f] = ACC_COUNT
    
final_data_svqa = pd.DataFrame(all_results['svqa']).T
final_data_mvqa = pd.DataFrame(all_results['mvqa']).T

final_data_svqa.to_excel("/online1/gzs_data/Personal_file/LiZhou/TemporalCultural/dataset/human_evaluation/human_svqa.xlsx")
final_data_mvqa.to_excel("/online1/gzs_data/Personal_file/LiZhou/TemporalCultural/dataset/human_evaluation/human_mvqa.xlsx")

print('a')
