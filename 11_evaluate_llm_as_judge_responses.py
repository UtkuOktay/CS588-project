import json
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from tqdm import tqdm

LLM_AS_JUDGE_RESPONSES_DIR = r'llm_as_judge_outputs'
MERGED_PRS_FILE_PATH = r'pr_diffs_filtered.jsonl'

RESULTS_SAVE_DIR = r'final_results'

os.makedirs(RESULTS_SAVE_DIR, exist_ok=True)

def get_pr_url_language_dict(file_path):
    with open(file_path, 'r') as f:
        data = [json.loads(line) for line in tqdm(f, desc="Reading PRs")]

    pr_url_language = dict()
    for pr in data:
        pr_url_language[pr['pr']['html_url']] = pr['language']

    return pr_url_language

def read_file(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)


pr_url_language_dict = get_pr_url_language_dict(MERGED_PRS_FILE_PATH)

files = [file for file in os.listdir(LLM_AS_JUDGE_RESPONSES_DIR) if file.endswith('.json')]

alignments = dict()

df_rows = []

for file in tqdm(files):
    judge_response = read_file(os.path.join(LLM_AS_JUDGE_RESPONSES_DIR, file))
    pr_url = judge_response['pr_url']
    language = pr_url_language_dict[pr_url]
    raw_llm_response = judge_response['llm_response']
    parsed_llm_response = json.loads(raw_llm_response)

    metrics = parsed_llm_response['metrics']
    system_b_coverage_of_a_ratio = metrics['system_b_coverage_of_a_ratio']
    system_a_coverage_of_b_ratio = metrics['system_a_coverage_of_b_ratio']
    overall_alignment_ratio = metrics['overall_alignment_ratio']

    if overall_alignment_ratio < 0 or overall_alignment_ratio > 1:
        continue

    df_rows.append((pr_url, language, overall_alignment_ratio, system_b_coverage_of_a_ratio, system_a_coverage_of_b_ratio))

df = pd.DataFrame(df_rows, columns=['pr_url', 'language', 'overall_alignment_ratio', 'system_b_coverage_of_a_ratio', 'system_a_coverage_of_b_ratio'])

for language in df['language'].unique():
    language_filtered_df = df[df['language'] == language]
    '''for column in ['overall_alignment_ratio', 'system_b_coverage_of_a_ratio', 'system_a_coverage_of_b_ratio']:
        print(f'{language} {column} stats => Mean: {language_filtered_df[column].mean()} - Median: {language_filtered_df[column].median()} - Min: {language_filtered_df[column].min()} - Max: {language_filtered_df[column].max()}')
    print('\n--------------------------------------------------\n')'''

    column = 'overall_alignment_ratio'
    print(
        f'{language} stats => Mean: {language_filtered_df[column].mean()} - Median: {language_filtered_df[column].median()} - Min: {language_filtered_df[column].min()} - Max: {language_filtered_df[column].max()}')

mean_order = (
    df.groupby('language')['overall_alignment_ratio']
    .mean()
    .sort_values(ascending=False)
    .index.tolist()
)

plt.clf()

sns.boxplot(x='language', y='overall_alignment_ratio', data=df, order=mean_order)
plt.title('Overall Alignment Ratio Distribution by Language')
plt.ylabel('Overall Alignment Ratio')
plt.xlabel('Language')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_SAVE_DIR, 'llm_as_judge_alignment_boxplot.png'))

plt.clf()

mean_df = df.groupby('language')['overall_alignment_ratio'].mean().reset_index()

sns.barplot(x='language', y='overall_alignment_ratio', data=mean_df, order=mean_order)

plt.title('Mean Overall Alignment Ratio by Language')
plt.ylabel('Mean Alignment Ratio')
plt.xlabel('Language')

ymin = mean_df['overall_alignment_ratio'].min() * 0.95
plt.ylim([mean_df['overall_alignment_ratio'].min() * 0.9, mean_df['overall_alignment_ratio'].min() * 1.1])

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_SAVE_DIR, 'llm_as_judge_alignment_mean_barplot.png'))

agg_df = df.groupby('language')['overall_alignment_ratio'].agg(['min', 'mean', 'max']).reset_index()
agg_df.to_html(os.path.join(RESULTS_SAVE_DIR, f'llm_as_judge_alignment_values.html'), index=False, float_format="{:.3f}".format)
