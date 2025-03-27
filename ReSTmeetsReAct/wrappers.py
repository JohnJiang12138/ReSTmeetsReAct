import json
import os
import gym
import numpy as np
import re
import string
from collections import Counter
import requests
import sys
import io
import time
import pdb

    
DATA_DIR = "data"
HOTPOTQA_SPLIT_FILE = {
  "train": "hotpot_train_v1.1_simplified.json",
  "dev": "hotpot_dev_v1_simplified.json",
  "test": "hotpot_test_v1_simplified.json",
}

FEVER_SPLIT_FILE = {
  "train": "train.jsonl",
  "dev": "paper_dev.jsonl",
}


class HistoryWrapper(gym.ObservationWrapper):
  def __init__(self, env, obs_format, prompt=None):
    super().__init__(env)
    assert obs_format in ["obs", "history"]
    if obs_format == "history":
      assert hasattr(self.env, "traj")
    self.obs_format = obs_format
    self.prompt = prompt if prompt is not None else ""

  def observation(self, obs):
    if self.obs_format == "obs":
      return obs
    elif self.obs_format == "history":
      observation = self.env.traj["observations"][0] + "\n"
      for i, (o, a) in enumerate(zip(self.env.traj["observations"][1:], self.env.traj["actions"]), 1):
        observation += f"Action {i}: {a}\nObservation {i}: {o}\n\n"
      return self.prompt + observation
    

def normalize_answer(s):
  def remove_articles(text):
    return re.sub(r"\b(a|an|the)\b", " ", text)
  
  def white_space_fix(text):
      return " ".join(text.split())

  def remove_punc(text):
      exclude = set(string.punctuation)
      return "".join(ch for ch in text if ch not in exclude)

  def lower(text):
      return text.lower()

  return white_space_fix(remove_articles(remove_punc(lower(s))))

url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/completions_pro?access_token=24.4b078140b720255d419b67f80c15c4ea.2592000.1716359237.282335-62758312"
def EB4(prompt,stop=None):
    stop_sequence = stop if stop is not None else []  # 确保stop参数是一个列表
    data = {
        "messages": [
            {"role": "user", "content": prompt},
        ],
        'max_tokens': 128,
        'frequency_penalty': 0.0,
        'presence_penalty': 0.0,
        "temperature": 0.8,
        "top_p": 0.75,
        'stop': stop_sequence  # 假设API支持类似的停止参数
    }

    headers ={
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }
    os.environ["http_proxy"] = ""
    os.environ["https_proxy"] = ""
    response = requests.request("POST", url, headers=headers, data=json.dumps(data))
    os.environ["http_proxy"] = "http://agent.baidu.com:8891"
    os.environ["https_proxy"] = "http://agent.baidu.com:8891"
    # print(response)
    # print(response.json())
    return response.json()['result']

def auto_eval(question, prediction, ground_truth, times = 1):
    prompt = f'''Check whether the ANSWER implies the REF_ANSWER to the question. 
    Here are some examples:
    ORIGINAL_QUESTION: Are McIntyre Mines and Copperfields Mine in the same country? ANSWER: So yes, they are in the same country. REF_ANSWER: yes.
    REPLY: True
    ORIGINAL_QUESTION: Are McIntyre Mines and Copperfields Mine in the same country? ANSWER: No, I don't think they are in the same country. REF_ANSWER: yes.
    REPLY: False
    ORIGINAL_QUESTION: {question}\nANSWER: {prediction}\nREF_ANSWER: {ground_truth}
    REPLY: 
    '''
    true_count = 0
    for _ in range(times):
        reply = EB4(prompt).lower()
        if reply=="true":
            true_count += 1
    
    return (float)(true_count/1.0)


def f1_score(prediction, ground_truth):
  normalized_prediction = normalize_answer(prediction)
  normalized_ground_truth = normalize_answer(ground_truth)

  ZERO_METRIC = (0, 0, 0)

  if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
    return ZERO_METRIC
  if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
    return ZERO_METRIC
  
  prediction_tokens = normalized_prediction.split()
  ground_truth_tokens = normalized_ground_truth.split()
  common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
  num_same = sum(common.values())
  if num_same == 0:
    return ZERO_METRIC
  precision = 1.0 * num_same / len(prediction_tokens)
  recall = 1.0 * num_same / len(ground_truth_tokens)
  f1 = (2 * precision * recall) / (precision + recall)
  return f1, precision, recall
  
class HotPotQAWrapper(gym.Wrapper):
  def __init__(self, env, split):
    super().__init__(env)
    data_file = f"{DATA_DIR}/{HOTPOTQA_SPLIT_FILE[split]}"
    self.data = json.load(open(data_file))
    self.data = [(d['question'], d['answer']) for d in self.data]
    self.data_idx = 0
    self.split = split

  def reset(self, seed=None, return_info=False, options=None, idx=None):
    self.env.reset(seed=seed, return_info=return_info, options=options)
    try:
      self.env.step('')
    except:
      pass
    self.env.reset(seed=seed, return_info=return_info, options=options)
    self.data_idx = int(np.random.randint(len(self.data))) if idx is None else idx
    observation = f"Question: {self.data[self.data_idx][0]}"
    info = self._get_info()
    return (observation, info) if return_info else observation

  def _get_info(self):
    return {
      "steps": self.steps, 
      "answer": self.answer,
      "question": self.data[self.data_idx][0], 
      "hotpot_split": self.split
    }

  def get_reward(self, info):
    if info['answer'] is not None:
      pred = normalize_answer(self.data[self.data_idx][1])
      gt = normalize_answer(info['answer'])
      score = (pred == gt)
      return int(score)
    return 0
  
  def get_metrics(self, info):
    if info['answer'] is not None:
      pred = normalize_answer(self.data[self.data_idx][1])
      gt = normalize_answer(info['answer'])
      question = self.data[self.data_idx][0] #同时获取原始问题，Auto-eval(question,answer,ref_answer)
      em = (pred == gt)#Exact Match
      f1 = f1_score(pred, gt)[0]
    #   auto_eval_score = auto_eval(question, pred, gt)
      return {'reward': em, 'em': em, 'f1': f1, 'auto_eval_score':0}
    return {'reward': 0, 'em': 0, 'f1': 0, 'auto_eval_score':0}

  def step(self, actions, thoughts=None,perplexities=None):
    if perplexities is None or not perplexities:
        raise ValueError("Perplexities cannot be None or empty for the decision process.")

    # 找到perplexities中最小值的索引
    min_index = perplexities.index(min(perplexities))

    # 从actions和thoughts中抽取对应最小perplexity的元素
    selected_action = actions[min_index]

    # 执行环境的步骤
    # TODO: first step obs does not have question. 
    obs, _, done, info = self.env.step(actions=actions,thoughts=thoughts,perplexities=perplexities)
    reward = self.get_reward(info)
    if done:
      obs = f"Episode finished, reward = {reward}\n"
      info.update({"gt_answer": self.data[self.data_idx][1], "question_idx": self.data_idx})
      info.update(self.get_metrics(info))
    return obs, reward, done, info
  
  def __len__(self):
    return len(self.data)

class LoggingWrapper(gym.Wrapper):
  def __init__(self, env, folder="trajs", file_id=None):
    super().__init__(env)
    self.trajs = []
    # self.traj = {"observations": [], "actions": [[]], "thoughts":[[]], "perplexities":[]}
    # 初始化actions和thoughts为包含四个空列表的二维列表
    self.traj = {
        "observations": [],
        "actions": [[], [], [], []],
        "thoughts": [[], [], [], []],
        "perplexities": [[], [], [], []]
    }
    self.folder = folder
    self.file_id = np.random.randint(0, 10000000) if file_id is None else file_id
    self.file_path = f"{self.folder}/{self.file_id}.json"
    os.makedirs("trajs", exist_ok=True)

  def __len__(self):
    return len(self.env.data)
  

  def reset(self, seed=None, return_info=False, options=None, idx=None):
    output = self.env.reset(seed=seed, return_info=return_info, options=options, idx=idx)
    observation = output[0] if return_info else output
    # self.traj = {"observations": [observation], "actions": [[]], "thoughts":[[]], "perplexities":[]}
    self.traj = {
        "observations": [observation],
        "actions": [[], [], [], []],
        "thoughts": [[], [], [], []],
        "perplexities": [[], [], [], []]
    }
    return output

  def step(self, actions, thoughts=None, perplexities=None):
    if perplexities is None or not perplexities:
        raise ValueError("Perplexities cannot be None or empty for the decision process.")

    # 找到perplexities中最小值的索引
    min_index = perplexities.index(min(perplexities))

    # 从actions和thoughts中抽取对应最小perplexity的元素
    selected_action = actions[min_index]
    selected_thought = thoughts[min_index] if thoughts else None

    # 执行环境的步骤
    obs, reward, done, info = self.env.step(actions=actions, thoughts=thoughts,perplexities=perplexities)
    
    # 更新traj数据结构
    self.traj["observations"].append(obs)
    
    # 将所有actions和thoughts存储在对应的列表中
    if actions and len(actions) > 0:
        for i, action in enumerate(actions):
            self.traj["actions"][i].append(action)

    if thoughts and len(thoughts) > 0:
        for i, thought in enumerate(thoughts):
            self.traj["thoughts"][i].append(thought)

    if perplexities and len(perplexities) > 0:
        for i, perplexity in enumerate(perplexities):
            self.traj["perplexities"][i].append(perplexity)
    if done:
        self.traj.update(info)

    return obs, reward, done, info


#   def step(self, actions,thoughts=None,perplexities=None):

#     obs, reward, done, info = self.env.step(action,thought=thought)
#     self.traj["observations"].append(obs)
#     self.traj["actions"].append(action)
#     self.traj["thoughts"].append(thought)
#     if done:
#       self.traj.update(info)
#     return obs, reward, done, info

  def update_record(self):
    if len(self.traj) > 0:
      self.trajs.append(self.traj)
      self.traj = {"observations": [], "actions": [[], [], [], []], "thoughts":[[], [], [], []], "perplexities":[[], [], [], []]}
  
  def write(self):
    self.update_record()
    with open(self.file_path, "w") as f:
      json.dump(self.trajs, f)
      print(f"Saved trajs to trajs/{self.file_id}.json")
    
  def close(self):
    self.write()

# class FeverWrapper(gym.Wrapper):
#   def __init__(self, env, split):
#     super().__init__(env)
    
#     data_path = f"./data/{FEVER_SPLIT_FILE[split]}"
#     with open(data_path, "r") as json_file:
#       json_list = list(json_file)

#     data = []
#     for json_str in json_list:
#       json_str = json.loads(json_str)
#       label = json_str["label"]
#       claim = json_str["claim"]
#       data.append((claim, label))

#     self.data = data
#     self.data_idx = 0
#     self.split = split

#   def reset(self, seed=None, return_info=False, options=None, idx=None):
#     self.env.reset(seed=seed, return_info=return_info, options=options)
#     try:
#       self.env.step('')
#     except:
#       pass
#     self.env.reset(seed=seed, return_info=return_info, options=options)
#     self.data_idx = int(np.random.randint(len(self.data))) if idx is None else idx
#     observation = f"Claim: {self.data[self.data_idx][0]}"
#     info = self._get_info()
#     return (observation, info) if return_info else observation

#   def _get_info(self):
#     return {
#       "steps": self.steps, 
#       "answer": self.answer,
#       "question": self.data[self.data_idx][0], 
#       "fever_split": self.split
#     }

#   def get_reward(self, info):
#     if info['answer'] is not None:
#       label = normalize_answer(self.data[self.data_idx][1])
#       pred = normalize_answer(info['answer'])
#       if label == pred:
#         return 1
#     return 0

#   def step(self, action,thought=None):
#     # TODO: first step obs does not have question. 
#     obs, _, done, info = self.env.step(action,thought=None)
#     reward = self.get_reward(info)
#     if done:
#       obs = f"Episode finished, reward = {reward}\n"
#       info.update({"gt_answer": self.data[self.data_idx][1], "question_idx": self.data_idx})
#       info.update({'em': reward, 'reward': reward, 'f1': reward})
#     return obs, reward, done, info
    
#   def __len__(self):
#     return len(self.data)
  
  

