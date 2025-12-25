import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
from tqdm import tqdm

import constants

MERGED_PRS_FILE_PATH = r"pr_diffs_filtered.jsonl"
RESPONSE_SAVE_FOLDER_PATH = r"llm_responses"

MODEL_NAME = "gpt-4.1-mini-2025-04-14"
NUM_WORKERS = 40
MINIMUM_RESPONSE_TIME = 10  # Wait until elapsed time reaches this duration. This is to make the duration predictable.
MAX_RETRIES = 2

os.makedirs(RESPONSE_SAVE_FOLDER_PATH, exist_ok=True)



client = OpenAI(api_key=constants.OPENAI_API_KEY)

PROMPT_TEMPLATE = """
# Context
You are an automated bot that reviews the provided pull requests (PRs) like a human developer. Your aim is to give a clear, actionable, respectful and unambiguous feedback. You should focus on both the code quality and its alignment with best practices, readability and maintainability. Your response should not be robotic or too formal. Do not be wordy, provide concise outputs. Avoid beginning your response with terms like "sure", "here is..." etc.
You will receive the code diff. provided in the pull request and the PR description. Especially focus on the code quality, readability, maintainability, testing, performance, security and suggestions. Provide a clear, comprehensive feedback and finally give your opinion on whether it should be merged.
# Output Format (must be strictly valid JSON):
```json
{
    "comments": [
        "1. <comment>",
        "2. <comment>"
    ],
    "should_be_merged": "<yes|no>"
}
```
# Details Regarding the Pull Request:
## Code Diff.:
{diff}

## PR Title:
{title}

## PR Description:
{desc}

# Reminder
Provide a high quality PR review, in terms of accuracy, comprehensiveness and clarity. Your response needs to mimic a human reviewer. Be as concise as possible.
"""

def read_file(file_path):
    with open(file_path, 'r') as f:
        pr_list = [json.loads(line) for line in tqdm(f, desc="Reading file:")]

    return pr_list

# -------------------------------------------------------------
# Call GPT with retry
# -------------------------------------------------------------
def call_gpt(prompt):
    for attempt in range(MAX_RETRIES):
        try:
            response = client.responses.create(
                model=MODEL_NAME,
                input=prompt,
            )
            return {
                'review_output': response.output_text,
                'input_tokens': response.usage.input_tokens,
                'cached_input_tokens': response.usage.input_tokens_details.cached_tokens,
                'output_tokens': response.usage.output_tokens

            }
        except Exception as e:
            print(f"\n⚠ GPT error: {e} (retry {attempt+1}/{MAX_RETRIES})")
            time.sleep(300)

    return None

def perform_inference(pr_record):
    start_time = time.time()

    language = pr_record['language']
    pr_id = pr_record['pr_id']

    if pr_id == 40:
        print()

    pr = pr_record['pr']
    pr_number = pr['pr_number']
    title = pr['title']
    description = pr['description']
    diff = pr['diff']

    description = 'No description provided' if description is None else description

    output_file_path = os.path.join(RESPONSE_SAVE_FOLDER_PATH, f'{pr_id}_{language}_{pr_number}.json')
    if os.path.exists(output_file_path):
        print(f'Skipping PR with ID {pr_id} because output file {output_file_path} already exists.')
        return

    prompt = PROMPT_TEMPLATE.replace("{title}", title).replace("{desc}", description).replace("{diff}", diff)
    response_attributes = call_gpt(prompt)
    if response_attributes is None:
        print(f"❌ Failed for PR {pr_id}, skipping.")
        return

    review_output = response_attributes['review_output']
    input_tokens = response_attributes['input_tokens']
    cached_input_tokens = response_attributes['cached_input_tokens']
    output_tokens = response_attributes['output_tokens']


    tmp_path = output_file_path + '.tmp'

    with open(tmp_path, 'w') as f:
        json.dump(
            {
                'pr_id': pr_id,
                'language': language,
                'pr_number': pr_number,
                'prompt': prompt,
                'review': review_output,
                'input_tokens': input_tokens,
                'cached_input_tokens': cached_input_tokens,
                'output_tokens': output_tokens,
            },
            f,
            indent=4,
        )

    os.replace(tmp_path, output_file_path)

    elapsed_time = time.time() - start_time
    time_to_sleep = MINIMUM_RESPONSE_TIME - elapsed_time
    time_to_sleep = 0 if time_to_sleep < 0 else time_to_sleep
    time.sleep(time_to_sleep)


if __name__ == '__main__':
    pr_list = read_file(MERGED_PRS_FILE_PATH)

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = [executor.submit(perform_inference, pr_record) for pr_record in pr_list]

        for _ in tqdm(as_completed(futures), total=len(futures), desc='Processing PRs'):
            pass
