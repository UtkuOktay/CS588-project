import copy
import json
import os
import time

import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from comment_cleaning import clean_comment

INPUT_FILES_DIR = r'pr_comments_grouped_and_filtered_based_on_selected_prs'
OUTPUT_FILES_DIR = r'pr_comments_with_meaningfulness_label'
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gemma3:4b"

os.makedirs(OUTPUT_FILES_DIR, exist_ok=True)

session = requests.Session()
adapter = requests.adapters.HTTPAdapter(pool_connections=20, pool_maxsize=20)
session.mount('http://', adapter)

def classify_comment(text):
    prompt = f"""
    Classify this pull-request comment strictly as either:

    MEANINGFUL – contains technical reasoning, criticism, explanation, code insight, or discussion.
    TRIVIAL – approvals, politeness, thanks, praise, emojis, vague statements, unclear feedback.

    Only output exactly: MEANINGFUL or TRIVIAL.

    Comment:
    {text}
    """
    try:
        payload = {
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,  # Get the whole answer at once
            "options": {
                "num_predict": 5,  # Limit tokens to prevent the model from "rambling"
            }
        }

        # Use the session instead of requests.post
        response = session.post(OLLAMA_URL, json=payload)
        response_text = response.json().get("response")
        response_text = response_text.replace("\n", " ").strip().upper()

        if 'MEANINGFUL' in response_text:
            return 'MEANINGFUL'

        if 'TRIVIAL' in response_text:
            return 'TRIVIAL'

        print(f'Response cannot be parsed: {response_text}. Text: {text}')
        return 'TRIVIAL'

    except Exception as e:
        print(f'Text: {text} - Exception: {e}')
        return None



def process_all_files():
    files = [f for f in os.listdir(INPUT_FILES_DIR) if f.endswith(".json")]

    for file_name in files:
        file_path = os.path.join(INPUT_FILES_DIR, file_name)
        with open(file_path, "r") as f:
            data = json.load(f)

        for pr_url, comments in tqdm(data.items(), desc=file_name):
            filtered_prs = dict()

            filename_suffix = f'_PR_{pr_url}'.replace('https://github.com/', '').replace('/pull/', '-').replace('/', '-')
            save_file_path = os.path.join(OUTPUT_FILES_DIR, file_name.replace('grouped_and_filtered_prs_', '').replace('.json', '') + filename_suffix) + '.json'

            if os.path.exists(save_file_path):
                print(f'File {save_file_path} already exists. Skipping.')
                continue

            for comment in comments:
                comment_text = comment['body']
                cleaned_comment_text = clean_comment(comment_text)
                label = classify_comment(cleaned_comment_text)
                if label is None:
                    continue

                try:
                    new_comment = copy.deepcopy(comment)
                    new_comment['meaningfulness_label'] = label

                    pr_comments = filtered_prs.get(pr_url, [])
                    pr_comments.append(new_comment)
                    filtered_prs[pr_url] = pr_comments

                    tmp_path = save_file_path + '.tmp'

                    with open(tmp_path, "w") as f:
                        json.dump(filtered_prs, f, indent=4)

                    os.replace(tmp_path, save_file_path)

                except Exception as e:
                    print(f'Exception in loop: {e}')


if __name__ == "__main__":
    process_all_files()