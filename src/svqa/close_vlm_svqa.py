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


parser.add_argument("--root_dir", default="/data1/home/lizhou/project/TemporalCultural", type=str)
parser.add_argument("--api_key", default="sk-7ldxnC50jJ1tnT1r1aA4F171843a4880B4B5238bE29eC462", type=str,)
parser.add_argument("--model_name", default="deepseek-v3", type=str,)
parser.add_argument('--instruction', nargs='+', type=str, help='List of instruction')
parser.add_argument("--face_info", action="store_true")
parser.add_argument("--extra_info", action="store_true")
args = parser.parse_args()

save_dir = args.root_dir + "/results/svqa/" + args.model_name
# 检查目录是否存在，如果不存在则创建
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

save_file = args.root_dir + "/results/svqa/" + args.model_name + '.json'

api_key = args.api_key
api_base = "https://api.ai-gaochao.cn/v1"
client = OpenAI(api_key=api_key, base_url=api_base)



specific_model = {
    'gpt-4o': "gpt-4o-2024-11-20",
    'deepseek-r1': "deepseek-r1",
    'doubao1-5-v': "doubao-1-5-vision-pro-32k",
    'gemini-25-p': "gemini-2.5-pro-exp-03-25",
}


for template in args.instruction:
    print(f"instruction: {template}")
    save_file = f"{save_dir}/{args.model_name}_{template}.json"
    instruction_file = f'dataset/instruction/svqa/{template}.txt'
    with open(os.path.join(args.root_dir, instruction_file), 'r') as files:
        instruction_prompt = files.readlines()
        instruction_prompt = "".join(instruction_prompt)


    if args.extra_info:
        with open(os.path.join(args.root_dir, 'dataset/merged_sivqa_info.json'), 'r', encoding='utf-8') as file:
            dataset = json.load(file)
    else:
        with open(os.path.join(args.root_dir, 'dataset/merged_sivqa.json'), 'r', encoding='utf-8') as file:
            dataset = json.load(file)



    data_output = []
    erros_count = 0
    for data in tqdm(dataset):
        if args.face_info:
            image_path = args.root_dir + '/dataset/raw_image/' + data['img_list'][0]
        else:
            image_path = args.root_dir + '/dataset/mask_image/' + data['img_list'][0]
        if "en" in template:
            final_promts = instruction_prompt + "\n" + "Question:"+data['base_question_en'] + "\n" + "Options: "+data['choices_en']
        else:
            final_promts = instruction_prompt + "\n" + "问题："+data['base_question'] + "\n" + "选项："+data['choices']

        if args.extra_info and "info" in data.keys():
            final_promts = final_promts + "\n补充信息："+ data['info']
        try:
            with open(image_path, "rb") as image_file:
                image_data = image_file.read()
            image_stream = io.BytesIO(image_data)
            base64_image = base64.b64encode(image_stream.getvalue()).decode()

            completion = client.chat.completions.create(
                model=specific_model[args.model_name],
                messages=[
                    {
                        "role": "user",
                        "content": [
                            # 文本部分
                            {"type": "text", "text": final_promts},
                            # 图片部分
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=1000
            )
        

            content = completion.choices[0].message.content
            data['predict'] = content
            data_output.append(data)
            with open(save_file, 'w', encoding='utf-8') as f:
                json.dump(data_output, f, ensure_ascii=False, indent=4)
        except Exception as e:

            erros_count = erros_count + 1
            print(data['question_id'])




    with open(save_file, 'w', encoding='utf-8') as f:
        json.dump(data_output, f, ensure_ascii=False, indent=4)

    print(f'error:{erros_count}')