import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import requests
import os
url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/completions_pro?access_token=24.4b078140b720255d419b67f80c15c4ea.2592000.1716359237.282335-62758312"

rank_prompt = '''Rater Instructions: - The goal of this rating is to filter out bad actions, so that they’ll be excluded from the fine-tuning dataset. - Overall, we want to the agent to produce relevant and grounded answers with minimal steps. Anything that deviates from this goal is considered bad. - If any element (thoughts, comments etc.) is empty, then it’s automatically bad. ######################################### *** Model Can See: ‘‘‘ {inputs} ‘‘‘ *** Model Output #1: ‘‘‘ {action1} ‘‘‘ *** Model Output #2: ‘‘‘ {action2} ‘‘‘ *** Model Output #3: ‘‘‘ {action3} ‘‘‘ *** Model Output #4: ‘‘‘ {action4} ‘‘‘ ######################################### Your Instructions: - Choose the best model output based on the rater’s instructions. - Don’t assume in your decision that the model knows anything outside of "Model Can See" section. 
- Be specific in your explanation. 
Output 3 lines(Explanation, Answer, Ranking) when answering and make sure to follow the precise format. 
Strictly follow the following format:
###Explanation: why you think model output #X is the best 
###Answer: #X 
###Ranking: #X > #Y > #Z > #M '''

def EB4(prompt, stop=None):
    stop_sequence = [stop] if stop is not None else []  # Ensure stop parameter is a list
    data = {
        "messages": [
            {"role": "user", "content": prompt},
        ],
        'max_tokens': 512,
        'frequency_penalty': 0.0,
        'presence_penalty': 0.0,
        "temperature": 0.8,
        "top_p": 0.5,
        'stop': stop_sequence
    }

    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }
    os.environ["http_proxy"] = ""
    os.environ["https_proxy"] = ""
    response = requests.post(url, headers=headers, json=data)
    os.environ["http_proxy"] = "http://agent.baidu.com:8891"
    os.environ["https_proxy"] = "http://agent.baidu.com:8891"
    return response.json()['result']
# 困惑度计算函数
def calculate_perplexity(text, model, tokenizer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)  # Ensure model is on the correct device

    # Encode text
    encodings = tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**encodings, labels=encodings["input_ids"])
        log_likelihood = outputs[0].item()

    # Convert the log likelihood to a tensor before applying exp
    log_likelihood_tensor = torch.tensor(-log_likelihood / encodings["input_ids"].shape[1], device=device)
    ppl = torch.exp(log_likelihood_tensor).item()
    return ppl

def load_model_and_predict(model_path, data_path, num_samples=1, output_file='output_predictions.txt'):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(model_path)
    model.to(device)
    model.eval()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Load data
    with open(data_path, 'r') as file:
        data = json.load(file)[:num_samples]  # Only load the first num_samples of data

    # Prepare output file
    with open(output_file, 'w') as f:
        for item in data:
            inputs = tokenizer.encode(item['input'], return_tensors="pt").to(device)
            input_length = inputs.shape[1]
            
            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    max_length=input_length + 256,
                    # num_beams=4,
                    # early_stopping=True,
                    do_sample=True,
                    num_return_sequences=4,
                    top_k=50,
                    temperature=0.5
                )

            output_texts = []
            for i, output in enumerate(outputs):
                generated_tokens = output[input_length:]
                predicted_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                output_texts.append(predicted_text)
                ppl = calculate_perplexity(predicted_text, model, tokenizer)
                f.write(f"Generated text {i+1}: {predicted_text}\n")
                f.write(f"Perplexity of generated text {i+1}: {ppl}\n")
                print(f"Generated text {i+1}: {predicted_text}")
                print(f"Perplexity of generated text {i+1}: {ppl}")

            # Format rank_prompt
            formatted_prompt = rank_prompt.format(inputs=   ['input'], action1=output_texts[0], action2=output_texts[1], action3=output_texts[2], action4=output_texts[3])
            result = EB4(formatted_prompt)  # Call EB4 with the formatted prompt
            f.write(formatted_prompt + "\n")
            f.write("EB4 Result: " + str(result) + "\n")
            print("EB4 Result: ", result)

# Paths to model and data
model_ckpt_path = "/root/paddlejob/workspace/env_run/jiangwenyuan/ReSTmeetsReAct/ReAct/model_checkpoints/best_model.ckpt"
data_json_path = "/root/paddlejob/workspace/env_run/jiangwenyuan/ReSTmeetsReAct/ReAct/valid_data.json"

# Call the function
load_model_and_predict(model_ckpt_path, data_json_path)

# import torch
# import json
# from transformers import AutoTokenizer, AutoModelForCausalLM
# rank_prompt = f'''Rater Instructions: - The goal of this rating is to filter out bad actions, so that they’ll be excluded from the fine-tuning dataset. - Overall, we want to the agent to produce relevant and grounded answers with minimal steps. Anything that deviates from this goal is considered bad. - If any element (thoughts, comments etc.) is empty, then it’s automatically bad. ######################################### *** Model Can See: ‘‘‘ {inputs} ‘‘‘ *** Model Output #1: ‘‘‘ {action1} ‘‘‘ *** Model Output #2: ‘‘‘ {action2} ‘‘‘ *** Model Output #3: ‘‘‘ {action3} ‘‘‘ *** Model Output #4: ‘‘‘ {action4} ‘‘‘ ######################################### Your Instructions: - Choose the best model output based on the rater’s instructions. - Don’t assume in your decision that the model knows anything outside of "Model Can See" section. - Be specific in your explanation. Output 3 lines when answering and make sure to follow the precise format. Explanation: why you think model output #X is the best Answer: #X Ranking: #X > #Y > ... '''
# def load_model_and_predict(model_path, data_path, num_samples=1, output_file='output_predictions.txt'):
#     # 设定设备
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
#     # 加载模型
#     model = AutoModelForCausalLM.from_pretrained(model_path)
#     model.to(device)
#     model.eval()
    
#     # 加载分词器
#     tokenizer = AutoTokenizer.from_pretrained(model_path)
    
#     # 加载数据
#     with open(data_path, 'r') as file:
#         data = json.load(file)[:num_samples]  # 仅加载前num_samples条数据

#     # 准备保存文件
#     with open(output_file, 'w') as f:
#         # 推理和打印输出
#         for item in data:
#             inputs = tokenizer.encode(item['input'], return_tensors="pt").to(device)
#             input_length = inputs.shape[1]  # 获取输入的token数量
            
#             with torch.no_grad():
#                 # 使用 generate 方法进行文本生成
#                 outputs = model.generate(
#                     inputs,
#                     max_length=input_length + 128,  # 设置最大长度为输入长度加上新生成的最大token数
#                     num_beams=4,    # 使用beam search
#                     early_stopping=True,  # 如果生成的句子达到逻辑上的结尾就停止
#                     do_sample=True,  # 开启随机采样
#                     num_return_sequences=4,  # 每个输入生成4个输出
#                     top_k=50,  # 从50个最有可能的词中采样
#                     temperature=0.5  # 设置temperature以控制生成的多样性
#                 )
            
#             # 记录输入与输出
#             f.write(f"Input: {item['input']}\n")
#             for i, output in enumerate(outputs):
#                 generated_tokens = output[input_length:]  # 去除输入部分
#                 predicted_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
#                 f.write(f"Output {i+1}: {predicted_text}\n")
#             f.write("\n")

# # 假设模型和数据路径
# model_ckpt_path = "/root/paddlejob/workspace/env_run/jiangwenyuan/ReSTmeetsReAct/ReAct/model_checkpoints/best_model.ckpt"
# data_json_path = "/root/paddlejob/workspace/env_run/jiangwenyuan/ReSTmeetsReAct/ReAct/valid_data.json"

# # 调用函数
# load_model_and_predict(model_ckpt_path, data_json_path)


