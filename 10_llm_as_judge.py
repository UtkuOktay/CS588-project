import json
import os.path

import pandas as pd
import requests
from pandas.core.computation.ops import isnumeric
from pydantic import BaseModel
from tqdm import tqdm

PR_DIFFS_FILE_PATH = r"pr_diffs_filtered.jsonl"

HUMAN_PR_COMMENTS_PATH = r'meaningful_pr_comments_merged.json'
LLM_PR_COMMENTS_PATH = r'parsed_llm_responses.json'

OUTPUT_DIR = r'llm_as_judge_outputs'

os.makedirs(OUTPUT_DIR, exist_ok=True)

session = requests.Session()
adapter = requests.adapters.HTTPAdapter()
session.mount('http://', adapter)

def read_json(file_path):
    with open(file_path) as json_file:
        return json.load(json_file)

def get_pr_url_diff_dict(pr_list):
    pr_url_diff_dict = dict()
    for pr in pr_list:
        pr_url_diff_dict[pr['pr']['html_url']] = pr['pr']['diff']

    return pr_url_diff_dict

def get_pr_language_num_human_comments_df(pr_list, human_pr_reviews):
    df_1 = pd.DataFrame([(pr['pr']['html_url'], pr['language']) for pr in pr_list], columns=['pr_url', 'language'])
    df_2 = pd.DataFrame([(pr_url, len(reviews)) for pr_url, reviews in human_pr_reviews.items()], columns=['pr_url', 'num_reviews'])

    return df_1.merge(df_2, on='pr_url')

class ExampleOutputSummaryCountsModel(BaseModel):
    matches_count: int
    unique_to_a_count: int
    unique_to_b_count: int

class ExampleOutputMetricsModel(BaseModel):
    system_b_coverage_of_a_ratio: float
    system_a_coverage_of_b_ratio: float
    overall_alignment_ratio: float

class ExampleOutput(BaseModel):
    summary_counts: ExampleOutputSummaryCountsModel
    metrics: ExampleOutputMetricsModel

def find_alignment(diff, human_comments, llm_comments):
    prompt = """
### ROLE
You are an objective technical auditor. You are comparing two automated code review systems (System A and System B). Your goal is to identify technical alignment and calculate coverage ratios.

### SOURCE MATERIAL: CODE DIFF
{code_diff}

### INPUT DATA: FINDINGS
#### [System A]:
{human_comments}
#### [System B]:
{llm_comments}

### STRICT DEFINITIONS
1. ALIGNED FINDING (MATCH): Both systems identified the same technical issue or suggestion. They do not need to use the same words, but they must point to the same logic/line and have the same intent.
2. UNIQUE FINDING: An issue mentioned by one system that is NOT mentioned by the other.
3. SYSTEM A: An automated review tool.
4. SYSTEM B: An automated review tool.

### EXECUTION STEPS
1. Step 1: Scan every item in System A. Check if it is matched by any item in System B.
2. Step 2: Scan every item in System B. Check if it is matched by any item in System A.
3. Step 3: Count the total number of unique points raised across both systems (Total = Matches + Unique_A + Unique_B).
4. Step 4: Calculate the ratios.

### CONSTRAINTS
- Ignore tone, politeness, or verbosity.
- If System A uses a code snippet and System B uses text, treat them as equivalent if they describe the same logic.
- Do not assume System A is "more correct" than System B.
- OUTPUT MUST BE VALID JSON.

### OUTPUT FORMAT
{
  "summary_counts": {
    "matches_count": [integer],
    "unique_to_a_count": [integer],
    "unique_to_b_count": [integer]
  },
  "metrics": {
    "system_b_coverage_of_a_ratio": [Matches / (Matches + Unique_A)],
    "system_a_coverage_of_b_ratio": [Matches / (Matches + Unique_B)],
    "overall_alignment_ratio": [Matches / (Matches + Unique_A + Unique_B)]
  }
}
"""

    prompt = prompt.replace('{code_diff}', diff).replace('{human_comments}', json.dumps(human_comments, indent=4)).replace('{llm_comments}', json.dumps(llm_comments, indent=4))

    try:
        payload = {
            "model": "gemma3:4b",
            "prompt": prompt,
            "stream": False,  # Get the whole answer at once
            "options": {
                "temperature": 0.0
            },
            "format": ExampleOutput.model_json_schema()
        }

        # Use the session instead of requests.post
        response = session.post("http://localhost:11434/api/generate", json=payload)
        response_text = response.json().get("response")

        return response_text

    except Exception as e:
        print(f'Exception: {e}')
        return None

def is_valid_response(response_text):
    try:
        response_text = response_text.strip().replace('```json', '').replace('```', '').upper()
        response_json = json.loads(response_text)
    except Exception:
        return False

    return True

human_pr_reviews = read_json(HUMAN_PR_COMMENTS_PATH)
llm_pr_reviews = read_json(LLM_PR_COMMENTS_PATH)

with open(PR_DIFFS_FILE_PATH, 'r') as f:
    pr_list = [json.loads(line) for line in tqdm(f, desc="Reading file:")]

pr_url_diff_dict = get_pr_url_diff_dict(pr_list)

pr_language_num_human_comments_df = get_pr_language_num_human_comments_df(pr_list, human_pr_reviews)

for language in pr_language_num_human_comments_df['language'].unique():
    num_processed_prs = 0

    filtered_pr_language_num_human_comments_df = pr_language_num_human_comments_df[(pr_language_num_human_comments_df['language'] == language) & (pr_language_num_human_comments_df['num_reviews'] <= 15)].sort_values('num_reviews', ascending=False)
    pr_url_list = filtered_pr_language_num_human_comments_df['pr_url'].tolist()
    all_pr_urls = set(human_pr_reviews.keys()).intersection(set(llm_pr_reviews.keys()))

    for pr_url in tqdm(pr_url_list):
        if pr_url not in llm_pr_reviews:
            continue

        diff = pr_url_diff_dict[pr_url]

        human_reviews = human_pr_reviews[pr_url]
        human_comments = [{'diff_hunk': human_review['diff_hunk'], 'comment_text': human_review['body']} for human_review in human_reviews]

        llm_comments = llm_pr_reviews[pr_url]['parsed_comments']

        response_text = find_alignment(diff, human_comments, llm_comments)

        save_filename = f'{language}_LLM_as_judge_{pr_url}'.replace('https://github.com/', '').replace('/pull/', '-').replace('/', '-') + '.json'
        response_file_save_path = os.path.join(OUTPUT_DIR, save_filename)
        tmp_file_path = response_file_save_path + '.tmp'

        save_object = {
                'pr_url': pr_url,
                'humman_comments': human_comments,
                'llm_comments': llm_comments,
                'llm_response': response_text
            }

        with open(tmp_file_path, 'w') as f:
            json.dump(save_object, f, indent=4)

        os.replace(tmp_file_path, response_file_save_path)








