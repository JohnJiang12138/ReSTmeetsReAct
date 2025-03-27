import json

# Load the data from the provided JSON files

with open("./trajs_EB4.json", "r", encoding="utf-8") as file:
    data_EB4 = json.load(file)
with open("./trajs_V0.json", "r", encoding="utf-8") as file:
    data_V0 = json.load(file)

with open("./trajs_V1.json", "r", encoding="utf-8") as file:
    data_V1 = json.load(file)

with open("./trajs_V2.json", "r", encoding="utf-8") as file:
    data_V2 = json.load(file)

# Extract the last 50 entries from V0

data_EB4_last_50 = data_EB4[-50:]
data_V0_last_50 = data_V0[-50:]

# V1 has exactly 50 entries as per requirement
data_V1_full = data_V1
data_V2_full = data_V2

# Convert to jsonl format and save
EB4_jsonl_path = "./trajs_EB4_last_50.jsonl"
v0_jsonl_path = "./trajs_V0_last_50.jsonl"
v1_jsonl_path = "./trajs_V1_full.jsonl"
v2_jsonl_path = "./trajs_V2_full.jsonl"

with open(EB4_jsonl_path, "w", encoding="utf-8") as file:
    for entry in data_EB4_last_50:
        file.write(json.dumps(entry) + "\n")

with open(v0_jsonl_path, "w", encoding="utf-8") as file:
    for entry in data_V0_last_50:
        file.write(json.dumps(entry) + "\n")

with open(v1_jsonl_path, "w", encoding="utf-8") as file:
    for entry in data_V1_full:
        file.write(json.dumps(entry) + "\n")

with open(v2_jsonl_path, "w", encoding="utf-8") as file:
    for entry in data_V2_full:
        file.write(json.dumps(entry) + "\n")

EB4_jsonl_path, v0_jsonl_path, v1_jsonl_path, v2_jsonl_path
# Define a function to process EB4 entries, not using perplexities
def process_eb4_entry(entry):
    # Get the last thought and action
    last_thought = entry['thoughts'][-1]
    last_action = entry['actions'][-1]
    
    # Combine the last thought and action
    combined = f"{last_thought} {last_action}"
    return combined
import csv

# Load JSONL data from the files previously saved
def load_jsonl(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return [json.loads(line) for line in file]
data_EB4 = load_jsonl("./trajs_EB4_last_50.jsonl")
data_V0 = load_jsonl("./trajs_V0_last_50.jsonl")
data_V1 = load_jsonl("./trajs_V1_full.jsonl")
data_V2 = load_jsonl("./trajs_V2_full.jsonl")
# Extract the last 50 entries from EB4 and V0
data_EB4_last_50 = data_EB4[-50:]
# Define a function to process perplexities, thoughts, and actions
def process_entry(entry):
    # Extract the minimum perplexity index from the last elements of each subarray in perplexities
    min_index = min(range(len(entry['perplexities'])), key=lambda x: entry['perplexities'][x][-1])
    
    # Get the last thought and action using the min_index
    last_thought = entry['thoughts'][min_index][-1]
    last_action = entry['actions'][min_index][-1]
    
    # Combine the last thought and action
    combined = f"{last_thought} {last_action}"
    return combined

# Prepare to write to CSV
csv_path = "./combined_entries.csv"
with open(csv_path, "w", newline='', encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    # Write header
    writer.writerow(["Question", "EB4 Thoughts & Actions", "EB4 Answer", "V0 Thoughts & Actions", "V0 Answer", "V1 Thoughts & Actions", "V1 Answer",  "V2 Thoughts & Actions", "V2 Answer", "Ground Truth Answer"])
    
    for i in range(50):
        # Process V0 and V1 entries
        eb4_thoughts_actions = process_eb4_entry(data_EB4_last_50[i])
        # print('eb4:',eb4_thoughts_actions)
        v0_thoughts_actions = process_entry(data_V0[i])
        v1_thoughts_actions = process_entry(data_V1[i])
        v2_thoughts_actions = process_entry(data_V2[i])
        
        # Write the row
        writer.writerow([
            data_V0[i]["observations"][0],  # Observations from V0 (same as V1 ideally)
            eb4_thoughts_actions,
            data_EB4_last_50[i]["answer"],
            v0_thoughts_actions,  # Processed thoughts & actions from V0
            data_V0[i]["answer"],  # Answer from V0
            v1_thoughts_actions,  # Processed thoughts & actions from V1
            data_V1[i]["answer"],  # Answer from V1
            v2_thoughts_actions,  # Processed thoughts & actions from V2
            data_V2[i]["answer"],  # Answer from V2
            data_V1[i]["gt_answer"]  # Ground truth answer from V1 (same as V0 ideally)
        ])

csv_path
import pandas as pd

# Load the CSV file into a DataFrame
df = pd.read_csv("./combined_entries.csv")

# Save the DataFrame to an Excel file
xlsx_path = "./combined_entries.xlsx"
df.to_excel(xlsx_path, index=False)

xlsx_path
