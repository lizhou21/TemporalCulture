import json
ann_file = "./annotations.json"
with open(ann_file, "r", encoding="utf-8") as f:
    ann = json.load(f)
# print(ann['1000']['meta'])

 # 替换1
QTYPES = {
    # 'type': ['汉元素服饰', '传统汉服形制', '汉服改良版'],  # 免得修改后面代码了
    # 'period': ['male', 'female'],
    # 'period': ['秦汉时期','魏晋时期', '唐朝', '宋朝', '明朝'],  # this is ordered, needs to be preserved
    # 'period': ['窄袖', '直袖', '半袖', '琵琶袖', '垂胡袖', '大袖'],
    # 'period': ['大襟', '对襟', '绕襟'],
    # 'period': ['直领', '坦领', '圆领', '方领', '立领', '交领'],
    # 'period': ['破群', '裤', '马面裙', '褶裙'],
    'period': ['比甲', '半臂', '云肩', '褙子', '披帛', '披风']
}

## generation template questions type:形制种类  # 替换2
QUESTION_TEXTS = {
    'period': ["以下图片中外搭服饰种类属于<period>的是？",  # choices: 1P, 3!=P
               "以下图片中外搭服饰种类不属于<period>的是？", # choices: 1!=P, 3=P
               "以下图片中外搭服饰种类与其他图片不同的是？", # choices: 1!=P, 3=P
               ]
}

## taken period for an example
# group by period
from collections import defaultdict
ann_period_grouped = defaultdict(list)
for k, v in enumerate(ann):
    ann_period_grouped[ann[v]["meta"]["outerwear"]].append(ann[v])  # 替换3 替换meta后面的类型。把meta中确定period的元素按period进行分组, change 'period' to other types, like 'gender'

# for item in ann_period_grouped:
#   print(item)
#   print(ann_period_grouped[item])
#   print(len(ann_period_grouped[item]))
#   # break

import random
from tqdm import tqdm


def generate_period_question_t1(ann_period_grouped):
    # "以下图片中的服饰属于<period>的风格的有？",  # choices: 1P, 3!=P
    question_data = []
    for period in QTYPES["period"]:  # change period to others
        total_items = len(ann_period_grouped[period])
        print(period, total_items)
        # print("obtaining question for period: ", period, " total items: ", total_items)
        # get 1 item from this period as the answer
        # get 3 items from other period as candidates
        for idx, answer_item in tqdm(enumerate(ann_period_grouped[period])):
            if len(answer_item["img_list"]) == 0:
                continue
            answer = {"image": random.choice(answer_item["img_list"]), "meta": answer_item["meta"]}
            candidates = []
            # get 3 other items from other period
            other_periods = list(set(QTYPES["period"]) - set([period]))
            if len(other_periods) >= 3:
                use_other_periods = random.sample(other_periods, 3)
                for other_period in use_other_periods:
                    valid_items = [item for item in ann_period_grouped[other_period] if item["img_list"]]
                    if not valid_items:
                        continue  # 如果这个朝代里没有有效项，就换一个

                    cand_item = random.choice(valid_items)
                    candidates.append({"image": random.choice(cand_item["img_list"]), "meta": cand_item["meta"]})
            else:  # 剩余不够三种
                while len(candidates) < 3:  # 可能有空值
                    other_period = random.choice(other_periods)
                    cand_item = random.choice(ann_period_grouped[other_period])

                    if not cand_item["img_list"]:
                        continue  # 跳过没有图片的项

                    candidates.append({
                        "image": random.choice(cand_item["img_list"]),
                        "meta": cand_item["meta"]
                    })
            # # 收集所有非当前 period 的样本
            # other_items = []
            # for other_period in set(QTYPES["period"]) - set([period]):
            #     other_items.extend(ann_period_grouped[other_period])
            #
            # possible_items = [x for x in other_items if x["img_list"]]
            # # 随机选 3 个干扰项
            # for cand_item in random.sample(possible_items, 3):
            #     candidates.append({"image": random.choice(cand_item["img_list"]), "meta": cand_item["meta"]})

            assert len(candidates) == 3

            question_data.append({"answer": answer,
                                 "candidates": candidates,
                                 "answer_img": answer["image"],
                                 "candidate_imgs": [c["image"] for c in candidates],
                                 "question": QUESTION_TEXTS["period"][0].replace("<period>", period),
                                 "question_type": "outerwear", # 替换4
                                 "question_formular": "outerwear_t1"}) # 替换5
    return question_data


import random

def generate_period_question_t2t3(ann_period_grouped):
    question_data = []
    for period in QTYPES["period"]:
        total_items = len(ann_period_grouped[period])
        for i in range(0, total_items, 3):
            candidates = []
            valid_group = True
            for x in ann_period_grouped[period][i:i+3]:
                if not x["img_list"]:  # 跳过空 img_list
                    valid_group = False
                    break
                candidates.append({"image": random.choice(x["img_list"]), "meta": x["meta"]})
            if not valid_group or len(candidates) != 3:
                continue

            # 打乱候选项
            random.shuffle(candidates)

            # 获取一个其他朝代的 period，确保它里面有至少一项 img_list 不为空
            other_periods = list(set(QTYPES["period"]) - set([period]))
            random.shuffle(other_periods)

            answer = None
            for answer_period in other_periods:
                possible_items = [x for x in ann_period_grouped[answer_period] if x["img_list"]]
                if possible_items:
                    answer_item = random.choice(possible_items)
                    answer = {"image": random.choice(answer_item["img_list"]), "meta": answer_item["meta"]}
                    break

            if not answer:
                continue  # 如果找不到合适的 answer，则跳过该题目

            question_template = random.choice(QUESTION_TEXTS["period"][1:3])
            question_data.append({
                "answer": answer,
                "candidates": candidates,
                "answer_img": answer["image"],
                "candidate_imgs": [c["image"] for c in candidates],
                "question": question_template.replace("<period>", period),
                "question_type": "outerwear",  # 替换6
                "question_formular": "outerwear_t2t3"  # 替换7
            })
    return question_data


def formulate_option_answers(question_data):
    options = [0, 0, 0, 0]
    answer_option_idx = random.randint(0, 3)  # Choose a random index between 0 and 3

    # Place the answer at the randomly chosen position
    options[answer_option_idx] = question_data["answer_img"]

    # Place the candidate images in the remaining positions, preserving their original order
    candidates = question_data["candidate_imgs"]
    assert len(candidates) == 3

    remaining_positions = [i for i in range(4) if i != answer_option_idx]
    for i in range(3):  # We have 3 candidates
        options[remaining_positions[i]] = candidates[i]

    return options, answer_option_idx

def generate_period_questions(ann_period_grouped):
    qid = 0
    question_data = []
    generate_fns = [generate_period_question_t1,
            generate_period_question_t2t3]
    for fn in generate_fns:
        generated_questions = fn(ann_period_grouped)
        for q in generated_questions:
            clean_q  = {}
            clean_q["question_meta"] = q
            clean_q["question"] = q["question"]
            clean_q["options"], clean_q["answer_idx"] = formulate_option_answers(q)
            clean_q["qid"] = "mivqa_%d"%qid
            qid += 1
            question_data.append(clean_q)
    return question_data

# mivqa_period_t1 = generate_period_question_t1(ann_period_grouped)
# print("total number of mivqa questions: ", len(mivqa_period_t1))

# mivqa_period_t2t3 = generate_period_question_t2t3(ann_period_grouped)
# print("total number of mivqa questions: ", len(mivqa_period_t2t3))

mivqa_period_questions = generate_period_questions(ann_period_grouped)
print("total questions generated: ", len(mivqa_period_questions))

print(mivqa_period_questions[0]["question"], "\n",
      mivqa_period_questions[0]["options"], "\n",
      mivqa_period_questions[0]["answer_idx"], "\n",
      mivqa_period_questions[0]["question_meta"], "\n",)

# 保存为 JSON 文件
with open("./mivqa_outerwear_questions.json", "w", encoding="utf-8") as f:  # 替换8
    json.dump(mivqa_period_questions, f, ensure_ascii=False, indent=4)