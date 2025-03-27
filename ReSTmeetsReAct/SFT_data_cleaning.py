import json

# 加载数据文件
with open('./trajs/5801665.json', 'r') as file:
    data = json.load(file)

# 遍历数据，清理每个元素的 'thoughts'
for i in range(300):
    print(i)
    thoughts = data[i]['thoughts']
    cleaned_thoughts = []
    for j in range(len(thoughts)):
        # 检查是否以 'Thought {i}: ' 开头，并移除该部分
        try:
            if thoughts[j].startswith('Thought '):
                index = thoughts[j].find(': ')
                if index != -1:
                    # 添加处理后的内容到列表
                    cleaned_thoughts.append(thoughts[j][index + 2:])
            else:
                cleaned_thoughts.append(thoughts[j])
        except:
            thoughts[j]='None'
            cleaned_thoughts.append(thoughts[j])
    # 更新 'thoughts' 字段
    data[i]['thoughts'] = cleaned_thoughts

# 保存处理后的数据到新文件
with open('./trajs/5801665_cleaned.json', 'w') as file:
    json.dump(data, file, indent=4)
