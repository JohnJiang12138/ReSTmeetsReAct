import os
import requests
import json
import sys
import io
import time
import pdb
import re
os.environ["http_proxy"] = "http://agent.baidu.com:8891"
os.environ["https_proxy"] = "http://agent.baidu.com:8891"

# url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/completions_pro?access_token=24.1d0abffc69c0c9fb781e2ee46118b60c.2592000.1713337901.282335-57060509"#expired
#24.4b078140b720255d419b67f80c15c4ea.2592000.1716359237.282335-62758312
url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/completions_pro?access_token=24.4b078140b720255d419b67f80c15c4ea.2592000.1716359237.282335-62758312" #EB4.0
# url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/completions?access_token=24.4b078140b720255d419b67f80c15c4ea.2592000.1716359237.282335-62758312"
# api_keys = [
#     'sk-8FvVpQuBf46eGGr8BbCfF38bB3F64b69Aa40366e801f6271'
# ]
api_keys = [
    'sk-viyCtivavwvqvyZ4B71bC4A8DfBf40748820A6129f4fCdA5',
    'sk-DEc5WgFTjPRa2GD46e6dE90553814207889669E91bDb6a8e'
]

# def chatgpt(prompt, stop=None, max_retries=10, timeout=10):
#     stop_sequence = stop if stop is not None else []  # 确保stop参数是一个列表
#     for attempt in range(max_retries):
#         api_key = api_keys[attempt % len(api_keys)]  # Rotate through the API keys
#         headers = {
#             'Authorization': f'Bearer {api_key}',
#             'Content-Type': 'application/json'
#         }
#         data = {
#             'model': 'gpt-3.5-turbo-1106',
#             'temperature': 0,
#             'max_tokens': 100,
#             'top_p': 0,
#             'frequency_penalty': 0.0,
#             'presence_penalty': 0.0,
#             'messages': [
#                 {
#                     "role": "user",
#                     "content": prompt
#                 }
#             ],
#             'stop': stop_sequence  # 假设API支持类似的停止参数
#         }

#         try:
#             # response = requests.post('https://api.kwwai.top/v1/chat/completions', headers=headers, json=data, timeout=timeout, stream=True)
#             response = requests.post('https://hk.xty.app/v1/chat/completions', headers=headers, json=data, timeout=timeout, stream=True)
#             if response.status_code == 200:
#                 return response.json()["choices"][0]["message"]["content"]
#             else:
#                 print(f"Error: {response.status_code}, {response.text}")
#         except requests.Timeout:
#             print("Request timed out, retrying...")
#         except requests.RequestException as e:
#             return f"Network error: {e}"

#     return "Error: Maximum retries reached"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
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

# 设定模型路径
# model_ckpt_path = "/root/paddlejob/workspace/env_run/jiangwenyuan/ReSTmeetsReAct/ReAct/model_checkpoints/best_model.ckpt" #7B
# model_ckpt_path = "/root/paddlejob/workspace/env_run/jiangwenyuan/ReSTmeetsReAct/ReAct/model_checkpoints_13B/best_model.ckpt" #13B V0
# model_ckpt_path = "/root/paddlejob/workspace/env_run/jiangwenyuan/ReSTmeetsReAct/ReAct/model_checkpoints_13B_V1_2/best_model.ckpt" #13B V1
# model_ckpt_path = "/root/paddlejob/workspace/env_run/jiangwenyuan/ReSTmeetsReAct/ReAct/model_checkpoints_13B_V2/best_model.ckpt" #13B V2
# model = AutoModelForCausalLM.from_pretrained(model_ckpt_path)
# tokenizer = AutoTokenizer.from_pretrained(model_ckpt_path)

def Llama(model, tokenizer, prompt, max_length=512, temperature=0.5, top_p=0.85, stop_sequence=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_length = inputs['input_ids'].shape[1]
    adjusted_max_length = input_length + max_length

    with torch.no_grad():
        output = model.generate(
            inputs['input_ids'],
            max_length=adjusted_max_length,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=1
        )

    generated_text = tokenizer.decode(output[0][input_length:], skip_special_tokens=True)
    
    if stop_sequence:
        stop_index = generated_text.find(stop_sequence)
        if stop_index != -1:
            generated_text = generated_text[:stop_index]

    ppl = calculate_perplexity(generated_text, model, tokenizer)

    return generated_text, ppl


# def Llama(model, tokenizer, prompt, max_length=256, temperature=0.5, top_p=0.85, num_return_sequences=1, stop_sequence=None):
#     # 确保CUDA可用性
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.to(device).eval()

#     # 编码输入文本
#     inputs = tokenizer(prompt, return_tensors="pt").to(device)
#     input_length = inputs['input_ids'].shape[1]
#     # 计算生成文本的最大长度
#     adjusted_max_length = input_length + max_length

#     # 生成文本
#     with torch.no_grad():
#         outputs = model.generate(
#             inputs['input_ids'],
#             max_length=adjusted_max_length,
#             temperature=temperature,
#             top_p=top_p,
#             num_return_sequences=num_return_sequences
#         )

#     # 保存生成的文本及其困惑度
#     generated_texts = []
#     perplexities = []

#     for output in outputs:
#         generated_text = tokenizer.decode(output[input_length:], skip_special_tokens=True)
        
#         # 如果指定了停止序列，则在输出中查找并截断
#         if stop_sequence:
#             stop_index = generated_text.find(stop_sequence)
#             if stop_index != -1:
#                 generated_text = generated_text[:stop_index]

#         # 计算每个生成文本的困惑度
#         ppl = calculate_perplexity(generated_text, model, tokenizer)

#         generated_texts.append(generated_text)
#         perplexities.append(ppl)

#     return generated_texts, perplexities

def Llama_single(model, tokenizer, prompt, max_length=512, temperature=0.5, top_p=0.85, stop_sequence=None):
    # 确保CUDA可用性
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    # 编码输入文本
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_length = inputs['input_ids'].shape[1]
    # 计算生成文本的最大长度
    adjusted_max_length = inputs['input_ids'].shape[1] + max_length

    # 生成文本
    with torch.no_grad():
        outputs = model.generate(
            inputs['input_ids'],
            max_length=adjusted_max_length,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=1
        )

    # 解码生成的文本，仅包括生成部分
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated_text = generated_text[len(prompt):]  # 仅提取生成的部分，而不包括输入的 prompt
    # generated_tokens = outputs[input_length:]
    # predicted_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    # ppl = calculate_perplexity(predicted_text, model, tokenizer)

    # 如果指定了停止序列，则在输出中查找并截断
    if stop_sequence:
        stop_index = generated_text.find(stop_sequence)
        if stop_index != -1:
            generated_text = generated_text[:stop_index]

    return generated_text


# 使用示例
# prompt = "Tell me something about artificial intelligence."
# stop_sequence = "\n"  # 假设我们在遇到换行符时停止输出
# output = Llama7B(prompt, stop_sequence=stop_sequence)
# print("Generated Text:", output)


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
# def EB4(prompt,stop="\n"):
#     stop_sequence = stop if stop is not None else []  # 确保stop参数是一个列表
#     data = {
#         "messages": [
#             {"role": "user", "content": prompt},
#         ],
#         'max_tokens': 128,
#         'frequency_penalty': 0.0,
#         'presence_penalty': 0.0,
#         "temperature": 0.8,
#         "top_p": 0.75,
#         'stop': stop_sequence  # 假设API支持类似的停止参数
#     }

#     headers ={
#         'Content-Type': 'application/json',
#         'Accept': 'application/json'
#     }
#     os.environ["http_proxy"] = ""
#     os.environ["https_proxy"] = ""
#     response = requests.request("POST", url, headers=headers, data=json.dumps(data))
#     # response = requests.post(url, headers=headers, json=data)
#     os.environ["http_proxy"] = "http://agent.baidu.com:8891"
#     os.environ["https_proxy"] = "http://agent.baidu.com:8891"
#     print(response)
#     print(response.json())
#     # print(response.json()["choices"][0]["message"]["content"])
#     return response.json()['result']




import wikienv, wrappers
env = wikienv.WikiEnv()
env = wrappers.HotPotQAWrapper(env, split="dev")
env = wrappers.LoggingWrapper(env)

def step(env, actions, thoughts=None,perplexities=None):
    if perplexities is None or not perplexities:
        raise ValueError("Perplexities cannot be None or empty for the decision process.")

    # 找到perplexities中最小值的索引
    min_index = perplexities.index(min(perplexities))

    # 从actions和thoughts中抽取对应最小perplexity的元素
    selected_action = actions[min_index]
    attempts = 0
    while attempts < 10:
        try:
            return env.step(actions=actions,thoughts=thoughts,perplexities=perplexities)
        except requests.exceptions.Timeout:
            attempts += 1

import json
import sys
import urllib3
urllib3.disable_warnings()
s = requests.Session()
s.verify = False

folder = './prompts/'
prompt_file = 'prompts_naive.json'
with open(folder + prompt_file, 'r') as f:
    prompt_dict = json.load(f)

webthink_examples = prompt_dict['webthink_simple6']
instruction = """Solve a question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be three types: 
(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
(2) Lookup[keyword], which returns the next sentence containing keyword in the current passage.
(3) Finish[answer], which returns the answer and finishes the task.
Here are some examples.
"""
webthink_prompt = instruction + webthink_examples

# def webthink(idx=None, prompt=webthink_prompt, to_print=True):
#     question = env.reset(idx=idx)
#     if to_print:
#         print(idx, question)
#     prompt += question + "\n"
#     n_calls, n_badcalls = 0, 0
#     for i in range(1, 8):
#         n_calls += 1
#         # os.environ["http_proxy"] = ""
#         # os.environ["https_proxy"] = ""
#         try:
#             thought_action = EB4(prompt + f"Thought {i}:", stop=[f"\nObservation {i}:"])
#         except:
#             print('invalid thought_action, skipped.')
#             continue
#         # thought_action = chatgpt(prompt + f"Thought {i}:", stop=[f"\nObservation {i}:"])
#         # os.environ["http_proxy"] = "http://agent.baidu.com:8891"
#         # os.environ["https_proxy"] = "http://agent.baidu.com:8891"
#         try:
#             thought, action = thought_action.strip().split(f"\nAction {i}: ")
#         except:
#             print('ohh...', thought_action)
#             n_badcalls += 1
#             n_calls += 1
#             thought = thought_action.strip().split('\n')[0]
#             # os.environ["http_proxy"] = ""
#             # os.environ["https_proxy"] = ""
#             action = EB4(prompt + f"Thought {i}: {thought}\nAction {i}:", stop=[f"\n"]).strip()
#             # action = chatgpt(prompt + f"Thought {i}: {thought}\nAction {i}:", stop=[f"\n"]).strip()
#             # os.environ["http_proxy"] = "http://agent.baidu.com:8891"
#             # os.environ["https_proxy"] = "http://agent.baidu.com:8891"
#         obs, r, done, info = step(env, action[0].lower() + action[1:])
#         obs = obs.replace('\\n', '')
#         step_str = f"Thought {i}: {thought}\nAction {i}: {action}\nObservation {i}: {obs}\n"
#         prompt += step_str
#         if to_print:
#             print(step_str)
#         if done:
#             break
#     if not done:
#         obs, r, done, info = step(env, "finish[]")
#     if to_print:
#         print(info, '\n')
#     info.update({'n_calls': n_calls, 'n_badcalls': n_badcalls, 'traj': prompt})
#     return r, info
def webthink(idx=None, prompt=webthink_prompt, to_print=True):
    question = env.reset(idx=idx)
    if to_print:
        print(idx, question)
    prompt += question + "\n"
    n_calls, n_badcalls = 0, 0
    for i in range(1, 8):
        n_calls += 1
        # thought_action = EB4(prompt + f"Thought {i}:", stop=[f"\nObservation {i}:"])
        #Llama7B(prompt, stop_sequence=stop_sequence)
        thought_actions = []
        perplexities = []
        for j in range(1):
            generated_text = EB4(prompt = prompt + f"Thought {i}:", stop=f"\nObservation {i}:")
            thought_actions.append(generated_text)
            ppl=0
            perplexities.append(ppl)

        #此处从原来的单条文本thought_action，变为多条文本，以及对应的ppl列表
        min_index = perplexities.index(min(perplexities))
        selected_thought_action = thought_actions[min_index]
        #我们只选取其中ppl最小的，去continue trajectory，其余的送入step函数，进行保存

        #使用enumerate进行循环，j为索引，thought_action和ppl分别为当前项
        for j, (thought_action, ppl) in enumerate(zip(thought_actions, perplexities)):
            print(f"Traj {j + 1}: {thought_action}")  # j + 1如果你希望从1开始计数
            print(f"Perplexity: {ppl}")
        thoughts = []
        actions = []
        # try:
        #     # selected_thought, selected_action =selected_thought_action.strip().split(f"\nAction {i}: ")
        #     selected_thought, selected_action =re.split(r'Action \d+: ', selected_thought_action)
        #     # thought_actions[min_index]=selected_thought+ " Action {i}: " + selected_action
        # except:
        #     print('ohh...', selected_thought_action)
        #     n_badcalls += 1
        #     n_calls += 1
        #     selected_thought = selected_thought_action.strip().split('\n')[0]
        #     # selected_thought = selected_thought_action.strip().split("Action")[0]
        #     selected_action = Llama_single(model=model, tokenizer=tokenizer, prompt = prompt + f"Thought {i}: {selected_thought}\nAction {i}:", max_length=256, temperature=0.5, top_p=0.85,stop_sequence=f"\n").strip()
            # thought_actions[min_index]=selected_thought+ " Action {i}: " + selected_action
        for thought_action in thought_actions:
            try:
                thought, action = thought_action.strip().split(f"\nAction {i}: ")
                # thought, action = re.split(r'Action \d+: ', thought_action)
                thoughts.append(thought)
                actions.append(action)
                # thought, action = thought_action.strip().split(f"\nAction {i}: ")
                # thoughts, actions = thought_action.strip().split(f"\nAction {i}: ") for thought_action in thought_actions
            except:
                # print('ohh...', thought_action)
                n_badcalls += 1
                n_calls += 1
                try:
                    thought = thought_action.strip().split('\n')[0]
                    # action = EB4(prompt + f"Thought {i}: {thought}\nAction {i}:", stop=[f"\n"]).strip()
                    #这一步是为了在exception的情况下，获取action到底是什么
                    action = EB4(prompt = prompt + f"Thought {i}: {thought}\nAction {i}:",stop=f"\n").strip()
                    ppl=0
                    thoughts.append(thought)
                    actions.append(action)
                except:
                    thought = thought_action.strip().split('Action')[0]
                    action = EB4(prompt = prompt + f"Thought {i}: {thought}\nAction {i}:",stop=f"\n").strip()
                    ppl=0
                    thoughts.append(thought)
                    actions.append(action)
        selected_thought, selected_action = thoughts[min_index], actions[min_index]
        print('selected_thought = ',selected_thought)
        print('selected_action = ',selected_action)
        print('min_index = ',min_index)
        # obs, r, done, info = step(env, action[0].lower() + action[1:],thought=thought)
        # 将所有actions和thoughts存储并执行最小ppl对应的action
        try:
            obs, r, done, info = step(env,actions= [action[0].lower()+action[1:] for action in actions], thoughts=thoughts, perplexities=perplexities)
            obs = obs.replace('\\n', '')
            step_str = f"Thought {i}: {selected_thought}\nAction {i}: {selected_action}\nObservation {i}: {obs}\n"
            prompt += step_str
            if to_print:
                print(step_str)
            if done:
                break
        except:
            pass
        # obs, r, done, info = env.step(actions=thought_actions, thoughts=[ta.split(f"\nAction {i}: ")[0] for ta in thought_actions], perplexities=perplexities)
        
    if not done:
        obs, r, done, info = env.step(actions=["finish[]"], thoughts=[None], perplexities=[0])
        # obs, r, done, info = step(env, "finish[]",thought=None)
          # Assuming finish has minimal perplexity effect
    if to_print:
        print(info, '\n')
    info.update({'n_calls': n_calls, 'n_badcalls': n_badcalls, 'traj': prompt})
    return r, info
import random
import time
idxs = list(range(7405))
random.Random(233).shuffle(idxs)

rs = []
infos = []
old_time = time.time()
# for i in idxs[:500]:
for i in idxs[:550]:
    if i in idxs[:300]:
        continue
    # if i in idxs[250:300]:
    #     continue
    r, info = webthink(i, to_print=True)
    # print('info = ',info)
    rs.append(info['em'])
    infos.append(info)
    print(sum(rs), len(rs), sum(rs) / len(rs), (time.time() - old_time) / len(rs))
    print('-----------')
    print()
    env.write()


