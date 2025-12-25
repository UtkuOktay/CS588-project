import json
import os
import re

from tqdm import tqdm

INPUT_FILES_DIR = r'dataset_raw_json_files'
OUTPUT_FILES_DIR = r'pr_comments_grouped_and_filtered_based_on_selected_prs'

FILTERED_PR_DIFFS_FILE_PATH = r'pr_diffs_filtered.jsonl'

os.makedirs(OUTPUT_FILES_DIR, exist_ok=True)

with open(FILTERED_PR_DIFFS_FILE_PATH, "r") as f:
    filtered_pr_diffs = [json.loads(line) for line in f]

filtered_pr_diffs_urls = set([record['pr']['html_url'] for record in filtered_pr_diffs])

pr_regex = re.compile(r"/pull/(\d+)")

files = [file for file in os.listdir(INPUT_FILES_DIR) if file.endswith(".json")]
for file in files:
    with open(os.path.join(INPUT_FILES_DIR, file), "r") as f:
        data = json.load(f)

    language = file.replace('mined-comments-25stars-25prs-', '').replace('.json', '').lower()
    pr_dict = dict()

    for repo_name, comments in tqdm(data.items(), desc=f'Processing {language} comments'):
        for comment in comments:
            comment_url = comment['html_url']
            pr_number = pr_regex.search(comment_url).group(1)
            pr_url = f'{comment_url.split('/pull/')[0]}/pull/{pr_number}'

            if pr_url not in filtered_pr_diffs_urls:
                continue

            comments_in_pr = pr_dict.get(pr_url, [])
            comments_in_pr.append(comment)
            pr_dict[pr_url] = comments_in_pr

    with open(os.path.join(OUTPUT_FILES_DIR, f'grouped_and_filtered_prs_{language}.json'), 'w') as f:
        json.dump(pr_dict, f, indent=4)
