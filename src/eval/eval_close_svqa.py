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

parser = argparse.ArgumentParser()


parser.add_argument("--root_dir", default="/data1/home/lizhou/project/TemporalCultural", type=str)
parser.add_argument("--model", default="gpt-4o", type=str,)
args = parser.parse_args()

read_file = args.root_dir + "/results/svqa/" + args.model + '.json'

with open(read_file, 'r', encoding='utf-8') as file:
    dataset_eval = json.load(file)
for data in tqdm(dataset_eval):
    if args.model == 'gpt-4o':
        predict = data['predict'][7:-3]
        answer_predict = json.loads(predict)['答案']
    question_type = data['question_type']
    TYPE_COUNT[question_type] += 1
    TYPE_COUNT['overall'] += 1
    if answer_predict == data['answer']:
        RIGHT_COUNT[question_type] += 1
        RIGHT_COUNT['overall'] += 1

for k, v in ACC_COUNT.items():
    ACC_COUNT[k] = RIGHT_COUNT[k]/TYPE_COUNT[k]
print(ACC_COUNT)