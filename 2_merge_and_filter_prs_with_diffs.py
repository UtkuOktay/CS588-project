import json
import os

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from tqdm import tqdm

INPUT_FILES_FOLDER_PATH = r'pr_diffs'
OUTPUT_FILE_SAVE_PATH = r'pr_diffs_filtered.jsonl'

THRESHOLD = 15000

def read_files(folder_path):
    files = [file for file in os.listdir(folder_path) if file.endswith('.jsonl')]
    data = dict()

    for file in files:  # [:1]:
        with open(os.path.join(INPUT_FILES_FOLDER_PATH, file), 'r') as f:
            data[file.replace('_prs.jsonl', '')] = [json.loads(line) for line in tqdm(f)]

    return data

def filter_prs(data):
    filtered_data = dict()

    for language, prs in data.items():
        for pr in prs:
            if len(pr['diff']) <= THRESHOLD:
                prs_in_language = filtered_data.get(language, [])
                prs_in_language.append(pr)
                filtered_data[language] = prs_in_language

    return filtered_data

def find_mean(data):
    for language, prs in data.items():
        diff_lengths = np.array([len(pr['diff']) for pr in prs])
        print(f'{language} => Mean: {np.mean(diff_lengths)} - Num: {len(diff_lengths)}')

def merge_prs(data):
    pr_list = []
    num_prs = 0

    for language, prs in data.items():
        for pr in prs:
            pr_list.append({
                'language': language,
                'pr_id': num_prs,
                'pr': pr
            })
            num_prs += 1

    return pr_list

def save_merged_prs(pr_list):
    with open(OUTPUT_FILE_SAVE_PATH, 'w') as f:
        for pr in pr_list:
            json.dump(pr, f)
            f.write('\n')

    print('Merged PRs saved to ' + OUTPUT_FILE_SAVE_PATH)

data = read_files(INPUT_FILES_FOLDER_PATH)
print('Before Filtering:')
find_mean(data)

data = filter_prs(data)
print('\nAfter Filtering:')
find_mean(data)

merged_prs = merge_prs(data)
save_merged_prs(merged_prs)
