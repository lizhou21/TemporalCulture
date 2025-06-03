import json
from collections import defaultdict
import random
from tqdm import tqdm
# rule based generation functions
def generate_period_question_t1(ann_period_grouped):
    # "以下图片中的服饰属于<period>的风格的有？",  # choices: 1P, 3!=P
    question_data = []
    for period in QTYPES["period"]:
        total_items = len(ann_period_grouped[period])
        # print("obtaining question for period: ", period, " total items: ", total_items)
        # get 1 item from this period as the answer
        # get 3 items from other period as candidates
        for idx, answer_item in tqdm(enumerate(ann_period_grouped[period])):
            answer = {"image": random.choice(answer_item["img_list"]), "meta": answer_item["meta"]}
            candidates = []
            # get 3 other items from other period
            other_periods = random.sample(list(set(QTYPES["period"]) - set([period])), 3)
            for other_period in other_periods:
                cand_item = random.choice(ann_period_grouped[other_period])
                candidates.append({"image": random.choice(cand_item["img_list"]), "meta": cand_item["meta"]})
            assert len(candidates) == 3
            question_data.append({"answer": answer,
                                 "candidates": candidates,
                                 "answer_img": answer["image"],
                                 "candidate_imgs": [c["image"] for c in candidates],
                                 "question": QUESTION_TEXTS["period"][0].replace("<period>", period),
                                 "question_type": "period",
                                 "question_formular": "period_t1"})
    return question_data

def generate_period_question_t2t3(ann_period_grouped):
    question_data = []
    for period in QTYPES["period"]:
        total_items = len(ann_period_grouped[period])
        # take every 3 items from ann_period_grouped[period] as a group
        for i in range(0, total_items, 3):
            candidates = []
            for x in ann_period_grouped[period][i:i+3]:
                candidates.append({"image": random.choice(x["img_list"]), "meta": x["meta"]})
            # print(candidates)
            if len(candidates) != 3:
                continue
            # shuffle the candidate items
            random.shuffle(candidates)

            # get 1 item from other period as the answer item
            answer_period = random.choice(list(set(QTYPES["period"]) - set([period])))
            # get 1 item from this period as the answer
            answer_item = random.choice(ann_period_grouped[answer_period])
            answer = {"image": random.choice(answer_item["img_list"]), "meta": answer_item["meta"]}
            
            question_template = random.choice(QUESTION_TEXTS["period"][1:3])
            
            question_data.append({"answer": answer,
                                 "candidates": candidates,
                                 "answer_img": answer["image"],
                                 "candidate_imgs": [c["image"] for c in candidates],
                                 "question": question_template.replace("<period>", period),
                                 "question_type": "period",
                                 "question_formular": "period_t2t3"})
    return question_data

def generate_period_question_t4(ann_period_grouped):
    # "以下图片中属于<period>以后时期的服饰有？" choices: 1 newer than p, 3 older than p
    question_data = []
    for period in QTYPES["period"][1:-1]:
        cur_period_index = QTYPES["period"].index(period)
        answer_period_candidates = QTYPES["period"][cur_period_index + 1:] # answer period should be later than cur_period
        answer_period = random.choice(answer_period_candidates)
        total_items = len(ann_period_grouped[answer_period])
        # print("obtaining question for clothes after period: ", period, " total items: ", total_items)
        for idx, answer_item in tqdm(enumerate(ann_period_grouped[answer_period])):
            answer = {"image": random.choice(answer_item["img_list"]), "meta": answer_item["meta"]}
            # get 3 items from period before cur_period as candidates
            candidates = []
            period_before = QTYPES["period"][:cur_period_index + 1]
            candidate_periods = [random.choice(period_before) for _ in range(3)]
            # add answer candidate
            for cand_period in candidate_periods:
                cand_item = random.choice(ann_period_grouped[cand_period])
                candidates.append({"image": random.choice(cand_item["img_list"]), "meta": cand_item["meta"]})
            
            question_data.append({"answer": answer,
                                 "candidates": candidates,
                                 "answer_img": answer["image"],
                                 "candidate_imgs": [c["image"] for c in candidates],
                                 "question": QUESTION_TEXTS["period"][3].replace("<period>", period),
                                 "question_type": "period",
                                 "question_formular": "period_t4"})
    return question_data


def generate_period_question_t5(ann_period_grouped):
    # "以下图片中比<image>中的服饰更古老的有？"] # context: p, answer 1 older than p, 3 newer than p
    question_data = []
    for period in QTYPES["period"][1:-1]:
        cur_period_index = QTYPES["period"].index(period)
        context_item = random.choice(ann_period_grouped[period])
        context = {"image": random.choice(context_item["img_list"]), "meta": context_item["meta"]}
        
        total_items = len(ann_period_grouped[period])
        # print("obtaining question for clothes after period: ", period, " total items: ", total_items)
        # answer period should be older than cur_period
        answer_period_candidates = QTYPES["period"][:cur_period_index] # answer period should be older than cur_period (index-1)
        for answer_period in answer_period_candidates:
            for idx, answer_item in tqdm(enumerate(ann_period_grouped[answer_period])):
                answer = {"image": random.choice(answer_item["img_list"]), "meta": answer_item["meta"]}
                # get 3 items from period before cur_period as candidates
                candidates = []
                period_after = QTYPES["period"][cur_period_index+1:]
                candidate_periods = [random.choice(period_after) for _ in range(3)]
                # add answer candidate
                for cand_period in candidate_periods:
                    cand_item = random.choice(ann_period_grouped[cand_period])
                    candidates.append({"image": random.choice(cand_item["img_list"]), "meta": cand_item["meta"]})
                
                question_data.append({"answer": answer,
                                    "candidates": candidates,
                                    "answer_img": answer["image"],
                                    "candidate_imgs": [c["image"] for c in candidates],
                                    "question": QUESTION_TEXTS["period"][4],
                                    "question_type": "period",
                                    "question_formular": "period_t5",
                                    "context": context})
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
                     generate_period_question_t2t3,
                     generate_period_question_t4,
                     generate_period_question_t5]
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

ann_file = "../../dataset/annotations.json"
ann = json.load(open(ann_file, "r"))

QTYPES = {
    'type': ['汉元素服饰', '传统汉服形制', '汉服改良版'], 
    'gender': ['男', '女'], 
    'period': ['秦汉时期','魏晋时期', '唐朝', '宋朝', '明朝'],  # this is ordered, needs to be preserved
    'xiu': ['窄袖', '直袖', '半袖', '琵琶袖', '垂胡袖', '大袖'], 
    'jin': ['大襟', '对襟', '绕襟'], 
    'ling': ['直领', '坦领', '圆领', '方领', '立领', '交领'], 
    'bottoms': ['破群', '裤', '马面裙', '褶裙'], 
    'outerwear': ['比甲', '半臂', '云肩', '褙子', '披帛', '披风']
}

## generation template questions
QUESTION_TEXTS = {
    'period': ["以下图片中的服饰属于<period>的风格的有？",  # choices: 1P, 3!=P
               "以下图片中不属于<period>的服饰有？", # choices: 1!=P, 3=P
               "以下图片中风格与其他服饰不同的有？", # choices: 1!=P, 3=P
               "以下图片中属于<period>以后时期的服饰有？", # choices: answer 1 newer than p, 3 older than p
               "以下的服装中比以上图片中的服饰更古老的有？"] # context: p, answer 1 older than p, 3 newer than p
}

ann_period_grouped = defaultdict(list)
for k, v in enumerate(ann):
    ann_period_grouped[ann[v]["meta"]["period"]].append(ann[v])


mivqa_period_questions = generate_period_questions(ann_period_grouped)
print("total questions generated: ", len(mivqa_period_questions))