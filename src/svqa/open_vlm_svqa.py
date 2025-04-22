import requests
import torch
from PIL import Image
from io import BytesIO
import yaml
import os
import json
from tqdm import tqdm 
import base64
import io
from lmdeploy import pipeline, TurbomindEngineConfig
from lmdeploy.vl import load_image

from qwen_vl_utils import process_vision_info
from transformers.image_utils import load_image
# from modelscope import snapshot_download
from transformers import AutoTokenizer, AutoProcessor, AutoModel, Qwen2_5_VLForConditionalGeneration, AutoModelForVision2Seq
from transformers import MllamaForConditionalGeneration, AutoProcessor

# import sivqa_utils
# import utils
import argparse
from tqdm import tqdm 

class Evaluator(object):
    def __init__(self, args):
        self.args = args
        self.model_name = args.model_name
    
    def _load_model(self):
        if os.path.exists(self.args.model_dir):
            model_file = os.path.join(self.args.model_dir, self.model_name)
        else:
            model_file = self.args.model_name
                
        if args.model_name == "Qwen2.5-VL-7B-Instruct":

            processor = AutoProcessor.from_pretrained(model_file)
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_file, torch_dtype="auto", device_map="auto").to(args.device).eval()
        elif args.model_name == "Idefics3-8B-Llama3": # only English output
            processor = AutoProcessor.from_pretrained(model_file)
            model = AutoModelForVision2Seq.from_pretrained(model_file, torch_dtype=torch.bfloat16).to(args.device).eval()

        
        elif args.model_name == "InternVL2_5-8B":
            model = pipeline(model_file, backend_config=TurbomindEngineConfig(session_len=8192))
            processor = model
            
        elif args.model_name == "MiniCPM-o-2_6":
            model = AutoModel.from_pretrained('openbmb/MiniCPM-o-2_6', trust_remote_code=True, attn_implementation='sdpa', 
                                              torch_dtype=torch.bfloat16, cache_dir=os.environ["HF_HOME"]) # sdpa or flash_attention_2, no eager
            model = model.eval().cuda()
            processor = AutoTokenizer.from_pretrained('openbmb/MiniCPM-o-2_6', trust_remote_code=True, cache_dir=os.environ["HF_HOME"])
        
        elif args.model_name == "Llama-3.2-11B-Vision":
            model_id = "meta-llama/Llama-3.2-11B-Vision"

            model = MllamaForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto", cache_dir=os.environ["HF_HOME"],
            )
            processor = AutoProcessor.from_pretrained(model_id, cache_dir=os.environ["HF_HOME"])
            

        return model, processor
    
    def eval_question(self, data, model, processor, args, instruction_prompt, template):
        if args.face_info:
            image_path = args.root_dir + '/dataset/raw_image/' + data['img_list'][0]
        else:
            image_path = args.root_dir + '/dataset/mask_image/' + data['img_list'][0]
        if "en" in template:
            final_promts = instruction_prompt + "\n" + "Question:"+data['base_question_en'] + "\n" + "Options: "+data['choices_en']
        else:
            final_promts = instruction_prompt + "\n" + "问题："+data['base_question'] + "\n" + "选项："+data['choices']

        if args.model_name == 'Qwen2.5-VL-7B-Instruct':
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": final_promts},
                        {"type": "image","image": image_path}
                    ]
                }
            ]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(args.device)
            generated_ids = model.generate(**inputs,max_new_tokens=200)

            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            print(output_text)

        elif args.model_name == "Idefics3-8B-Llama3":
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": final_promts},
                        {"type": "image"}
                    ]
                }
            ]
            input_image = load_image(image_path)
            prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = processor(text=prompt, images=[input_image], return_tensors="pt")
            inputs = {k: v.to(args.device) for k, v in inputs.items()}
            # Generate
            generated_ids = model.generate(**inputs, max_new_tokens=500)
            output_text = processor.batch_decode(generated_ids, skip_special_tokens=True)

        elif args.model_name == "InternVL2_5-8B":
            image = load_image(image_path)
            response = model((final_promts, image))
            output_text = response.text
            
        elif args.model_name == "MiniCPM-o-2_6":
            image = Image.open(image_path).convert("RGB")
            question = final_promts
            msgs = [{'role': 'user', 'content': [image, question]}]

            output_text = model.chat(
                msgs=msgs,
                tokenizer=processor
            )
        
        elif args.model_name == "Llama-3.2-11B-Vision":
            image = Image.open(image_path).convert("RGB")
            prompt = "<|image|><|begin_of_text|>" + final_promts
            inputs = processor(image, prompt, return_tensors="pt").to(model.device)
            generated_ids = model.generate(**inputs, max_new_tokens=500)
            output_text = processor.batch_decode(generated_ids, skip_special_tokens=True)

        return output_text

            



def main(args):
    # load model and processor
    evaluator = Evaluator(args)
    
    model, processor = evaluator._load_model()
    save_dir = args.root_dir + "/results/svqa/" + args.model_name
    # 检查目录是否存在，如果不存在则创建
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


    for template in args.instruction:
        print(f"instruction: {template}")
        save_file = f"{save_dir}/{args.model_name}_{template}.json"
        instruction_file = f'dataset/instruction/svqa/{template}.txt'
        with open(os.path.join(args.root_dir, instruction_file), 'r') as files:
            instruction_prompt = files.readlines()
            instruction_prompt = "".join(instruction_prompt)

        # read_data
        with open(os.path.join(args.root_dir, 'dataset/merged_sivqa.json'), 'r', encoding='utf-8') as file:
            dataset = json.load(file)


        data_output = []
        erros_count = 0
        for data in tqdm(dataset):

            content = evaluator.eval_question(data, model, processor, args, instruction_prompt, template)
            data['predict'] = content
            data_output.append(data)
            with open(save_file, 'w', encoding='utf-8') as f:
                json.dump(data_output, f, ensure_ascii=False, indent=4)
        
        with open(save_file, 'w', encoding='utf-8') as f:
            json.dump(data_output, f, ensure_ascii=False, indent=4)

        print(f'error:{erros_count}')

                
    

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--root_dir", default="/online1/gzs_data/Personal_file/LiZhou/TemporalCultural")
    argparser.add_argument("--model_dir", default="/online1/gzs_data/LLM-models")
    argparser.add_argument("--model_name", default="Qwen2.5-VL-7B-Instruct")
    argparser.add_argument("--template", type=int, default=0)
    argparser.add_argument("--lang", default="zh")
    argparser.add_argument("--face_info", action="store_true")
    argparser.add_argument('--instruction', nargs='+', type=str, help='List of instruction')
    # argparser.add_argument("--thinking", action="store_true", default=False)
    
    args = argparser.parse_args()
    if torch.cuda.is_available():
        args.device = torch.device("cuda")
    else:
        args.device = torch.device("cpu")
    print(args.device)

    
    main(args)
    