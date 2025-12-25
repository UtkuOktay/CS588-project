import json
import os
import re

from tqdm import tqdm

INPUT_FILES_DIR = r'llm_responses'
OUTPUT_FILE_PATH = r'parsed_llm_responses.json'

MERGED_PRS_FILE_PATH = r"pr_diffs_filtered.jsonl"

def safe_json_loads(text):
    """
    Try to parse JSON that came back as a string from GPT.
    Also tries a simple repair for trailing commas.
    """
    if not isinstance(text, str):
        return None
    text = text.strip()
    if not text:
        return None

    text = text.replace('```json', '').replace('```', '')
    # First attempt: direct parse
    try:
        return json.loads(text)
    except Exception:
        pass

    # Second attempt: remove common trailing commas
    repaired = re.sub(r",\s*}", "}", text)
    repaired = re.sub(r",\s*]", "]", repaired)

    repaired = repaired.replace('\\', '\\\\')

    try:
        return json.loads(repaired)
    except Exception as e:
        return None


def strip_leading_number(comment: str) -> str:
    """
    Removes leading patterns like:
    '1. text', '2) text', '3 - text', '12.   text'
    """
    if not isinstance(comment, str):
        return comment
    return re.sub(r'^\s*\d+[\.\)\-]\s*', '', comment).strip()


def get_pr_id_url_dict(file_path):
    pr_id_url_dict = dict()

    with open(file_path, 'r') as f:
        pr_list = [json.loads(line) for line in tqdm(f, desc="Reading file:")]

    for pr in pr_list:
        pr_id_url_dict[pr['pr_id']] = pr['pr']['html_url']

    return pr_id_url_dict


files = [f for f in os.listdir(INPUT_FILES_DIR) if f.endswith(".json")]

review_dict = dict()

pr_id_url_dict = get_pr_id_url_dict(MERGED_PRS_FILE_PATH)
for file in tqdm(files):
    with open(os.path.join(INPUT_FILES_DIR, file), "r", encoding='utf-8') as input_file:
        data = json.load(input_file)

    raw_review_comments = data['review']

    parsed_comments = safe_json_loads(raw_review_comments)

    if parsed_comments is None:
        continue

    if isinstance(parsed_comments, dict):
        comments = parsed_comments.get("comments", [])
        if isinstance(comments, list):
            cleaned_comments = [strip_leading_number(c) for c in comments]
            parsed_comments["comments"] = cleaned_comments

    parsed_review = {
        'pr_id': data['pr_id'],
        'pr_url': pr_id_url_dict[data['pr_id']],
        'language': data['language'],
        'raw_review': data['review'],
        'parsed_comments': parsed_comments['comments']
    }

    review_dict[parsed_review['pr_url']] = parsed_review

with open(OUTPUT_FILE_PATH, 'w', encoding='utf-8') as output_file:
    json.dump(review_dict, output_file, indent=4)
