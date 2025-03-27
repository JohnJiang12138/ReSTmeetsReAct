import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model_and_predict(model_path, data_path, num_samples=10):
    # 设定设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(model_path)
    model.to(device)
    model.eval()
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # 加载数据
    with open(data_path, 'r') as file:
        data = json.load(file)[:num_samples]  # 仅加载前num_samples条数据

    # 推理和打印输出
    for item in data:
        inputs = tokenizer.encode(item['input'], return_tensors="pt").to(device)
        # inputs = tokenizer.encode("Hey, how are you? I am ", return_tensors="pt").to(device)
        input_length = inputs.shape[1]  # 获取输入的token数量
        
        with torch.no_grad():
            # 使用 generate 方法进行文本生成
            outputs = model.generate(
                inputs,
                max_length=input_length + 128,  # 设置最大长度为输入长度加上新生成的最大token数
                num_beams=2,    # 使用beam search，beams数量可以根据需要调整
                early_stopping=True  # 如果生成的句子达到逻辑上的结尾就停止
            )
        
        # 从输出中只取生成的新内容部分
        generated_tokens = outputs[0, input_length:]  # 假设输出中去除输入部分
        predicted_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        # print(f"Input: {item['input']}\nPredicted Output: {predicted_text}\n")
        print(f"Predicted Output: {predicted_text}\n")

# 假设模型和数据路径
# model_ckpt_path = "/root/paddlejob/workspace/env_run/jiangwenyuan/ReSTmeetsReAct/ReAct/model_checkpoints/best_model.ckpt"
model_ckpt_path = "/root/paddlejob/workspace/env_run/jiangwenyuan/ReSTmeetsReAct/ReAct/model_checkpoints_13B_V1/best_model.ckpt"
# model_ckpt_path = "/root/paddlejob/workspace/env_run/jiangwenyuan/ReSTmeetsReAct/ReAct/Llama-2-13b-chat-hf"
# model_ckpt_path = "/root/paddlejob/workspace/env_run/jiangwenyuan/ReSTmeetsReAct/ReAct/Llama-2-7b-hf"
data_json_path = "/root/paddlejob/workspace/env_run/jiangwenyuan/ReSTmeetsReAct/ReAct/valid_data.json"

# 调用函数
load_model_and_predict(model_ckpt_path, data_json_path)


# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM
# import json

# def load_model_and_predict(model_base_path, model_ckpt_path, data_path, num_samples=10):
#     # 设定设备
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
#     # 加载基础模型
#     model = AutoModelForCausalLM.from_pretrained(model_base_path)
    
#     # 加载检查点
#     model.load_state_dict(torch.load(model_ckpt_path, map_location=device))
#     model.to(device)
#     model.eval()
    
#     # 加载分词器
#     tokenizer = AutoTokenizer.from_pretrained(model_base_path)
    
#     # 加载数据
#     with open(data_path, 'r') as file:
#         data = json.load(file)[:num_samples]  # 仅加载前10条数据

#     # 推理和打印输出
#     for item in data:
#         inputs = tokenizer.encode(item['input'], return_tensors="pt").to(device)
#         attention_mask = torch.ones(inputs.shape, dtype=torch.long).to(device)  # 假设全关注
        
#         with torch.no_grad():
#             outputs = model.generate(inputs, attention_mask=attention_mask)
        
#         predicted_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
#         print(f"Input: {item['input']}\nPredicted Output: {predicted_text}\n")

# # 路径设定
# model_base_path = "/root/paddlejob/workspace/env_run/jiangwenyuan/ReSTmeetsReAct/ReAct/Llama-2-7b-hf"
# model_ckpt_path = "/root/paddlejob/workspace/env_run/jiangwenyuan/ReSTmeetsReAct/ReAct/model_checkpoints/best_model.ckpt"
# data_json_path = "/root/paddlejob/workspace/env_run/jiangwenyuan/ReSTmeetsReAct/ReAct/training_data.json"

# # 调用函数
# load_model_and_predict(model_base_path, model_ckpt_path, data_json_path)


# #多卡推理
# import torch
# import json
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from torch.nn.parallel import DistributedDataParallel as DDP
# import torch.distributed as dist
# import os

# def setup(rank, world_size):
#     os.environ['MASTER_ADDR'] = 'localhost'
#     os.environ['MASTER_PORT'] = '12355'
#     dist.init_process_group("nccl", rank=rank, world_size=world_size)

# def cleanup():
#     dist.destroy_process_group()

# def load_model_and_predict(model_path, data_path, num_samples=10, rank=0, world_size=1):
#     setup(rank, world_size)
    
#     # 设置设备
#     device = torch.device(f"cuda:{rank}")
    
#     # 加载模型
#     model = AutoModelForCausalLM.from_pretrained(model_path)
#     model.to(device)
#     model = DDP(model, device_ids=[rank])
#     model.eval()
    
#     # 加载分词器
#     tokenizer = AutoTokenizer.from_pretrained(model_path)
    
#     # 加载数据
#     with open(data_path, 'r') as file:
#         data = json.load(file)[:num_samples]  # 仅加载前num_samples条数据

#     # 推理和打印输出
#     for item in data:
#         inputs = tokenizer.encode(item['input'], return_tensors="pt").to(device)
#         input_length = inputs.shape[1]  # 获取输入的token数量
        
#         with torch.no_grad():
#             outputs = model.module.generate(
#                 inputs,
#                 max_length=input_length + 128,
#                 num_beams=2,
#                 early_stopping=True
#             )
        
#         generated_tokens = outputs[0, input_length:]
#         predicted_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
#         if rank == 0:
#             print(f"Predicted Output: {predicted_text}\n")
    
#     cleanup()

# # 启动多进程环境
# if __name__ == "__main__":
#     world_size = torch.cuda.device_count()  # 获取GPU数量
#     torch.multiprocessing.spawn(load_model_and_predict,
#                                 args=(model_ckpt_path, data_json_path, 10),
#                                 nprocs=world_size,
#                                 join=True)

