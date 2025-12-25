import copy
import json
import os

from tqdm import tqdm

INPUT_FILES_DIR = r'pr_comments_with_meaningfulness_label'
OUTPUT_FILE_PATH = r'meaningful_pr_comments_merged.json'

files = [f for f in os.listdir(INPUT_FILES_DIR) if f.endswith(".json")]

pr_dict = dict()

label_set = set()
for file in tqdm(files):
    with open(os.path.join(INPUT_FILES_DIR, file), "r", encoding='utf-8') as input_file:
        data = json.load(input_file)

    for pr_url, comments in data.items():
        filtered_comments = []

        for comment in comments:
            meaningfulness_label = comment['meaningfulness_label']
            label_set.add(meaningfulness_label)
            if meaningfulness_label.upper() != "MEANINGFUL":
                continue

            filtered_comment = copy.deepcopy(comment)
            filtered_comment.pop('meaningfulness_label')
            filtered_comments.append(filtered_comment)

        pr_dict[pr_url] = filtered_comments

with open(OUTPUT_FILE_PATH, 'w', encoding='utf-8') as output_file:
    json.dump(pr_dict, output_file, indent=4)
