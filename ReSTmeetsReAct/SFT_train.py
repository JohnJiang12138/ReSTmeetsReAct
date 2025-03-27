import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.optim import AdamW
from tqdm import tqdm
import json
import os
import argparse
from torch.nn.utils.rnn import pad_sequence

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

from torch.utils.data import Dataset
import json
import copy

class WebThinkDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=4096):
        with open(file_path, 'r') as file:
            self.data = json.load(file)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.bos_token = tokenizer.bos_token or tokenizer.cls_token
        self.eos_token = tokenizer.eos_token or tokenizer.sep_token

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        record = self.data[idx]
        input_text = record['input']
        output_text = record['output'] + self.eos_token

        # Encode the input and the output text
        encoded_input = self.tokenizer(input_text, add_special_tokens=False, return_tensors='pt')
        encoded_output = self.tokenizer(output_text, add_special_tokens=False, return_tensors='pt')
        
        # Combine input_ids and output_ids
        input_ids = torch.cat((encoded_input['input_ids'], encoded_output['input_ids']), dim=-1).squeeze(0)
        
        # Ensure the combined input_ids do not exceed max_length
        if input_ids.size(0) > self.max_length:
            input_ids = input_ids[:self.max_length]
        
        # Initialize labels to be ignored everywhere
        labels = torch.full(input_ids.shape, -100, dtype=torch.long)
        
        # Update labels where output tokens start
        output_start = encoded_input['input_ids'].size(1)
        output_end = output_start + encoded_output['input_ids'].size(1)
        
        # Adjust output_end if it exceeds max_length
        output_end = min(output_end, self.max_length)
        
        # Set labels for the output part
        labels[output_start:output_end] = encoded_output['input_ids'].squeeze(0)[:output_end-output_start]

        # Create an attention_mask for the input_ids
        attention_mask = torch.ones(input_ids.size(0), dtype=torch.long)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }


def custom_collate_fn(batch):
    # 批处理中的每个元素形如 {'input_ids': tensor, 'attention_mask': tensor, 'labels': tensor}
    input_ids = pad_sequence([item['input_ids'] for item in batch], batch_first=True, padding_value=0)
    attention_mask = pad_sequence([item['attention_mask'] for item in batch], batch_first=True, padding_value=0)
    labels = pad_sequence([item['labels'] for item in batch], batch_first=True, padding_value=-100)

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }

def train(rank, world_size, args):
    setup(rank, world_size)
    model = AutoModelForCausalLM.from_pretrained(args.model_path)
    
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    # tokenizer.add_special_tokens({'pad_token': 0})
    # print('tokenizer.pad_token_id = ',tokenizer.pad_token_id)
    # tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.add_special_tokens({'pad_token': '<PAD>'})
    # model.to(torch.bfloat16)
    model.train()
    # explore_model_structure(model)
    # Freeze all parameters initially
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the last five layers
    num_layers = 32  # Total number of layers
    layers_to_unfreeze = 1  # Number of last layers you want to unfreeze

    # Correctly accessing and unfreezing layers
    for i in range(num_layers - layers_to_unfreeze, num_layers):
        layer = getattr(model.model, 'layers')[i]  # Access layers by indexing directly
        for param in layer.parameters():
            param.requires_grad = True
    
    # 解冻最后一层的参数
    for param in model.lm_head.parameters():
        param.requires_grad = True

    # Optionally print out which parameters are trainable to confirm the setting
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(f"{name} is unfrozen and trainable.")
    #     else:
    #         print(f"{name} remains frozen.")


    torch.cuda.set_device(rank)
    model.to(rank)
    model = DDP(model, device_ids=[rank])
    dataset = WebThinkDataset(args.data_path, tokenizer)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    # dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler)
    # 使用 DataLoader 时指定 collate_fn：
    dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, collate_fn=custom_collate_fn)
    
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)

    # 初始化最佳损失为无限大
    best_loss = float('inf')
    for epoch in range(args.epochs):
        torch.cuda.empty_cache()
        sampler.set_epoch(epoch)
        total_loss = 0
        
        # 初始化 tqdm 只在 rank 0 的进程中
        loop = tqdm(dataloader, leave=True) if rank == 0 else dataloader
        
        for batch in loop:
            input_ids = batch['input_ids'].to(rank)
            attention_mask = batch['attention_mask'].to(rank)
            labels = batch['labels'].to(rank)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss.mean()

            optimizer.zero_grad()
            loss.backward()
            # 梯度剪裁
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            if rank == 0:
                loop.set_description(f"Epoch {epoch+1}")
                loop.set_postfix(loss=loss.item())
                total_loss += loss.item()

        if rank == 0:
            epoch_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch+1} average loss: {epoch_loss}")
            # 检查是否是最佳模型
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                # 保存为 .ckpt 文件
                checkpoint_path = os.path.join(args.save_path, f"best_model.ckpt")
                model.module.save_pretrained(checkpoint_path)
                print(f"Saved best model checkpoint to {checkpoint_path}")

    cleanup()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--model_path", type=str, default="/root/paddlejob/workspace/env_run/jiangwenyuan/ReSTmeetsReAct/ReAct/Llama-2-13b-chat-hf")
    parser.add_argument("--model_path", type=str, default="/root/paddlejob/workspace/env_run/jiangwenyuan/ReSTmeetsReAct/ReAct/model_checkpoints_7B_V0/best_model.ckpt")
    # parser.add_argument("--model_path", type=str, default="/root/paddlejob/workspace/env_run/jiangwenyuan/ReSTmeetsReAct/ReAct/Llama-2-7b-hf")
    # parser.add_argument("--model_path", type=str, default="/root/paddlejob/workspace/env_run/jiangwenyuan/ReSTmeetsReAct/ReAct/llama3/Meta-Llama-3-8B")
    # parser.add_argument("--data_path", type=str, default="/root/paddlejob/workspace/env_run/jiangwenyuan/ReSTmeetsReAct/ReAct/training_data_v2.json")
    parser.add_argument("--data_path", type=str, default="/root/paddlejob/workspace/env_run/jiangwenyuan/ReSTmeetsReAct/ReAct/training_data_v1_7B.json")
    parser.add_argument("--save_path", type=str, default="/root/paddlejob/workspace/env_run/jiangwenyuan/ReSTmeetsReAct/ReAct/model_checkpoints_7B_V1/")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=torch.cuda.device_count())
    args = parser.parse_args()
    torch.multiprocessing.spawn(train, args=(args.world_size, args), nprocs=args.world_size, join=True)
