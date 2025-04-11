import pandas as pd
import json
import random
import os
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

# 设置matplotlib显示中文
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans', 'sans-serif']  # 用来正常显示中文
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.rcParams['font.size'] = 12  # 设置字体大小

# 映射字段到question_type


CANDIDATE_ANSWER = {
    'type': ['汉元素服饰', '传统汉服形制', '汉服改良版'], 
    'gender': ['男', '女'], 
    'period': ['魏晋时期', '明朝', '宋朝', '秦汉时期', '唐朝'], 
    'xiu': ['窄袖', '直袖', '半袖', '琵琶袖', '垂胡袖', '大袖'], 
    'jin': ['大襟', '对襟', '绕襟'], 
    'ling': ['直领', '坦领', '圆领', '方领', '立领', '交领'], 
    'bottoms': ['破群', '裤', '马面裙', '褶裙'], 
    'outerwear': ['比甲', '半臂', '云肩', '褙子', '披帛', '披风']
}

TYPE_COUNT = {
    'type': 0,
    'gender': 0,
    'period': 0,
    'xiu': 0,
    'jin': 0,
    'ling': 0,
    'bottoms': 0,
    'outerwear': 0
}

# 为每个字段定义对应的问题文本
QUESTION_TEXTS = {
    'type': "图片中的服饰通常属于以下哪个类型？",
    'gender': "图片中的服饰通常适合什么性别？",
    'period': "图片中的服饰属于以下哪个时期的风格？",
    'xiu': "图片中服饰的袖子属于以下哪种类型？",
    'jin': "图片中服饰的襟型属于以下哪种类型？",
    'ling': "图片中服饰的领型属于以下哪种类型？",
    'bottoms': "图片中服饰的下身是什么类型的？",
    'outerwear': "图片中服饰的外搭是什么？"
}



def load_json_data(json_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def get_candidate_choice(original_list, answer, num_to_select=3):

    working_list = original_list.copy()
    
    working_list.remove(answer)

    if len(working_list) < num_to_select:
        other_choice =  working_list
    else:
        other_choice = random.sample(working_list, num_to_select)
    other_choice.append(answer)
    random.shuffle(other_choice)
    return other_choice

def get_choice(id2choices):
    choice = ""
    for i, (k, v) in enumerate(id2choices.items()):
        if i == 0:
            choice = choice+k+"."+v
        else:
            choice = choice+"；"+k+"."+v
    return choice

def main(json_file):
    

    data = load_json_data(json_file)
    fields_to_process = list(QUESTION_TEXTS.keys())
    
    all_questions = []
    question_id = 0

    # obtain candidate answer
    # candidate_answer = {}
    # for f in fields_to_process:
    #     candidate_answer[f]=[]
    # for d_id, d_info in data.items():
    #     for meta, info in d_info['meta'].items():
    #         if meta in fields_to_process:
    #             candidate_answer[meta].append(info)

    # for ty, anwers in candidate_answer.items():
    #     candidate_answer[ty] = set(anwers)

    for d_id, d_info in data.items():
        for meta, info in d_info['meta'].items():
            if meta in fields_to_process:
                q_d = {
                    "question_id": "single_"+str(question_id),
                    "question_type": meta,
                    "cloth_id": d_id,
                    "img_list": d_info["img_list"],
                    "base_question": QUESTION_TEXTS[meta]
                }
                if meta == 'gender':
                    if info != 'unsure':
                        gender2map={
                            "female":"女",
                            "male": "男"
                        }

                        id2choices = {
                            "A": "男",
                            "B": "女"
                        }
                        choice2id = {value: key for key, value in id2choices.items()}
                        choice = get_choice(id2choices)
                        q_d['choices']=choice
                        q_d['answer']=choice2id[gender2map[info]]
                        if len(d_info["img_list"])>0:
                            all_questions.append(q_d)
                            question_id = question_id + 1
                            TYPE_COUNT[meta] += 1
                else:
                    answer = info
                    candidate = CANDIDATE_ANSWER[meta]
                    if answer in candidate:
                        candidate_choice = get_candidate_choice(candidate,answer)
                        if len(candidate_choice) == 3:
                            id2choices = {
                                "A": candidate_choice[0],
                                "B": candidate_choice[1],
                                "C": candidate_choice[2]
                            }
                        elif len(candidate_choice) == 4:
                            id2choices = {
                                "A": candidate_choice[0],
                                "B": candidate_choice[1],
                                "C": candidate_choice[2],
                                "D": candidate_choice[3],
                            }
                        else:
                            print('a')

                        choice2id = {value: key for key, value in id2choices.items()}
                        choice = get_choice(id2choices)
                        q_d['choices']=choice
                        q_d['answer']=choice2id[answer]
                        if len(d_info["img_list"])>0:
                            all_questions.append(q_d)
                            question_id = question_id + 1
                            TYPE_COUNT[meta] += 1



    print(f"question number: {len(all_questions)}")
    print(TYPE_COUNT)
    
    if all_questions:
        merged_json_filename = "merged_sivqa.json"
        with open(merged_json_filename, 'w', encoding='utf-8') as f:
            json.dump(all_questions, f, ensure_ascii=False, indent=2)

    
if __name__ == "__main__":
    # 执行数据处理
    root_dir = '/online1/gzs_data/Personal_file/LiZhou/TemporalCultural'
    file_name = os.path.join(root_dir, 'dataset/annotations.json')
    main(file_name)