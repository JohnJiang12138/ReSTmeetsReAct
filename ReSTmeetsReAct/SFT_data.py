import json

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

# Load data file
with open('./trajs/5801665_cleaned.json', 'r') as file:
    data = json.load(file)

#trajs/3933413.json
with open('./trajs/3933413.json', 'r') as file:
    data_EB4 = json.load(file)

all_training_pairs = []
all_validation_pairs = []
# Iterate through all trajectories
for i in range(550):
    if i in range(300):
        thoughts = data[i]['thoughts']
        actions = data[i]['actions']
        observations = data[i]['observations']

        training_pairs = []
        current_input = webthink_prompt + observations[0] + '\n'  # Initial input

        for j in range(len(thoughts)):
            # Build the output
            output = f"Thought {j+1}: {thoughts[j]}\nAction {j+1}: {actions[j]}"
            training_pairs.append({'input': current_input, 'output': output})

            # Update the input for the next round
            if j + 1 < len(observations):
                current_input += f"Thought {j+1}: {thoughts[j]}\nAction {j+1}: {actions[j]} Observation {j+1}: {observations[j+1]} "

        # Splitting into training and validation sets
        if i < 250:
            all_training_pairs.extend(training_pairs)
        else:
            all_validation_pairs.extend(training_pairs)
    else:
        thoughts = data_EB4[i-300]['thoughts'][0]
        actions = data_EB4[i-300]['actions'][0]
        observations = data_EB4[i-300]['observations']

        training_pairs = []
        current_input = webthink_prompt + observations[0] + '\n'  # Initial input

        for j in range(len(thoughts)):
            # Build the output
            output = f"Thought {j+1}: {thoughts[j]}\nAction {j+1}: {actions[j]}"
            training_pairs.append({'input': current_input, 'output': output})

            # Update the input for the next round
            if j + 1 < len(observations):
                current_input += f"Thought {j+1}: {thoughts[j]}\nAction {j+1}: {actions[j]} Observation {j+1}: {observations[j+1]} "

        all_training_pairs.extend(training_pairs)

# Save the training data to a JSON file
with open('./training_data_500.json', 'w') as f:
    json.dump(all_training_pairs, f, ensure_ascii=False, indent=4)

# Save the validation data to a JSON file
with open('./valid_data.json', 'w') as f:
    json.dump(all_validation_pairs, f, ensure_ascii=False, indent=4)

print("Training data saved to './training_data_500.json'")
print("Validation data saved to './valid_data.json'")


# import json

# # 加载初始的指导和示例数据
# folder = './prompts/'
# prompt_file = 'prompts_naive.json'
# with open(folder + prompt_file, 'r') as f:
#     prompt_dict = json.load(f)
# webthink_examples = prompt_dict['webthink_simple6']
# instruction = """Solve a question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be three types: 
# (1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
# (2) Lookup[keyword], which returns the next sentence containing keyword in the current passage.
# (3) Finish[answer], which returns the answer and finishes the task.

# """
# webthink_prompt = instruction + '\n'

# # 加载数据文件
# with open('./trajs/5801665_cleaned.json', 'r') as file:
#     data = json.load(file)

# all_training_pairs = []

# # 遍历所有轨迹
# for i in range(300):
#     thoughts = data[i]['thoughts']
#     actions = data[i]['actions']
#     observations = data[i]['observations']

#     training_pairs = []
#     current_input = webthink_prompt + "\n" + observations[0] + '\n'  # 初始输入

#     for j in range(len(thoughts)):
#         # 构建输出
#         output = f"Thought {j+1}: {thoughts[j]} Action {j+1}: {actions[j]}"
#         training_pairs.append({'input': current_input, 'output': output})

#         # 更新输入为下一轮
#         if j + 1 < len(observations):
#             current_input += f"Thought {j+1}: {thoughts[j]} Action {j+1}: {actions[j]} Observation {j+1}: {observations[j+1]} "

#     all_training_pairs.extend(training_pairs)

# # 将所有训练对保存到JSON文件
# with open('./training_data.json', 'w') as f:
#     json.dump(all_training_pairs, f, ensure_ascii=False, indent=4)

# print("Training data saved to './training_data.json'")


# import json

# folder = './prompts/'
# prompt_file = 'prompts_naive.json'
# with open(folder + prompt_file, 'r') as f:
#     prompt_dict = json.load(f)
# webthink_examples = prompt_dict['webthink_simple6']
# instruction = """Solve a question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be three types: 
# (1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
# (2) Lookup[keyword], which returns the next sentence containing keyword in the current passage.
# (3) Finish[answer], which returns the answer and finishes the task.
# Here are some examples.
# """
# webthink_prompt = instruction + webthink_examples
# print('webthink_prompt = ')
# print(webthink_prompt)

# # Load the JSON file
# with open('./trajs/5801665.json', 'r') as file:
#     data = json.load(file)

# for i in range(300):
#     # Extract the sections
#     thoughts = data[i]['thoughts']
#     actions = data[i]['actions']
#     observations = data[i]['observations']

#     training_pairs = []
#     current_input = webthink_prompt + "\n" + observations[0] +'\n' # 初始输入

#     for j in range(len(thoughts)):
#         # 构建输出
#         output = f"\nThought {j+1}: {thoughts[j]}\nAction {j+1}: {actions[j]}\n"
#         training_pairs.append((current_input, output))

#         # 更新输入为下一轮
#         if i + 1 < len(observations):
#             current_input += f"\nThought {j+1}: {thoughts[j]}\nAction {j+1}: {actions[j]}\nObservation {j+1}: {observations[j+1]}\n"

#     # 输出构建的训练对
#     for pair in training_pairs:
#         print("Input:", pair[0])
#         print("Output:", pair[1])
#         print("---")


# Prepare the output string
# output = observations[0] + '\n'  # Start with the first observation
# max_len = max(len(thoughts), len(actions), len(observations) - 1)  # Find the longest list

# # Interleave the rest of the observations, thoughts, and actions
# for i in range(max_len):
#     if i < len(thoughts):
#         output += f"Thought {i+1}: " + thoughts[i] + '\n'
#     if i < len(actions):
#         output += f"Action {i+1}: " + actions[i] + '\n'
#     if i + 1 < len(observations):
#         output += f"Observation {i+1}: " + observations[i + 1] + '\n'

# print(output)
