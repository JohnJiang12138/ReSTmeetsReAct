import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import requests
import os
import re
url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/completions_pro?access_token=24.4b078140b720255d419b67f80c15c4ea.2592000.1716359237.282335-62758312" #EB4
# url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/completions?access_token=24.4b078140b720255d419b67f80c15c4ea.2592000.1716359237.282335-62758312" #EB3.5

traj_path = "/root/paddlejob/workspace/env_run/jiangwenyuan/ReSTmeetsReAct/ReAct/trajs/1447036.json"

# Load initial guidance and example data
folder = './prompts/'
prompt_file = 'prompts_naive.json'
with open(folder + prompt_file, 'r') as f:
    prompt_dict = json.load(f)
webthink_examples = prompt_dict['webthink_simple6']
instruction = """Solve a question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be three types: 
(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
(2) Lookup[keyword], which returns the next sentence containing keyword in the current passage.
(3) Finish[answer], which returns the answer and finishes the task.
"""
webthink_prompt = instruction

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



# def load_data(file_path):
#     with open(file_path, 'r') as file:
#         data = json.load(file)
#     return data
# def construct_prompt(thoughts, actions):
#     combined = ""
#     for t_list, a_list in zip(thoughts, actions):
#         for t, a in zip(t_list, a_list):
#             combined += f"Thought: {t} Action: {a}\n"
#     return combined
def extract_number_from_answer(text):
    number = re.findall(r'Answer: #(\d+)', text)
    return int(number[0])


# Load data file
# with open('./trajs/1447036.json', 'r') as file:
#     data = json.load(file)
#4212314
with open('./trajs/4212314.json', 'r') as file:
    data = json.load(file)

all_training_pairs = []
all_validation_pairs = []
# Iterate through all trajectories
for i in range(550):
    if i >=250 and i < 300:
        continue
    thoughts = data[i]['thoughts']
    actions = data[i]['actions']
    observations = data[i]['observations']
    perplexities = data[i]['perplexities']
    training_pairs = []
    current_input = webthink_prompt + observations[0] + '\n'  # Initial input
    min_len = min(min([len(t) for t in thoughts]),min(len(a) for a in actions))
    for j in range(min_len):
        # Format rank_prompt
        formatted_prompt = rank_prompt.format(inputs=current_input, action1=f"Thought: {thoughts[0][j]} Action: {actions[0][j]}\n", action2=f"Thought: {thoughts[1][j]} Action: {actions[1][j]}\n", action3=f"Thought: {thoughts[2][j]} Action: {actions[2][j]}\n", action4=f"Thought: {thoughts[3][j]} Action: {actions[3][j]}\n")
        # print('formatted_prompt = ',formatted_prompt)
        try:
            result = EB4(formatted_prompt)
            print('EB4 result: ',result)
            number = extract_number_from_answer(result)
            if number<0 or number>4:
                four_ppl = [perplexities[0][j],perplexities[1][j],perplexities[2][j],perplexities[3][j]]
                number = four_ppl.index(min(four_ppl)) + 1 #注意返回的需要加上1
        except:
            four_ppl = [perplexities[0][j],perplexities[1][j],perplexities[2][j],perplexities[3][j]]
            number = four_ppl.index(min(four_ppl)) + 1 #注意返回的需要加上1
        print('number = ', number)
        output = f"Thought {j+1}: {thoughts[number-1][j]}\n Action {j+1}: {actions[number-1][j]}"
        training_pairs.append({'input': current_input, 'output': output})
        # Update the input for the next round
        if j + 1 < len(observations):
            current_input += f"Thought {j+1}: {thoughts[number-1][j]}\n Action {j+1}: {actions[number-1][j]} Observation {j+1}: {observations[j+1]} "
        # Splitting into training and validation sets
    if i < 550:
        all_training_pairs.extend(training_pairs)
    else:
        all_validation_pairs.extend(training_pairs)
    
# Save the training data to a JSON file
with open('./training_data_v2_7B.json', 'w') as f:
    json.dump(all_training_pairs, f, ensure_ascii=False, indent=4)

# Save the validation data to a JSON file
# with open('./valid_data_v2.json', 'w') as f:
#     json.dump(all_validation_pairs, f, ensure_ascii=False, indent=4)

print("Training data saved to './training_data_v2_7B.json'")
print("Validation data saved to './valid_data_v2_7B.json'")

