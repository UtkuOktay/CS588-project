import json
import os
import pickle

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

import constants
from comment_cleaning import clean_comment

HUMAN_PR_COMMENTS_PATH = r'meaningful_pr_comments_merged.json'
LLM_PR_COMMENTS_PATH = r'parsed_llm_responses.json'

CACHE_FILE_NAME = r'concat_embeddings.pkl'

RESULTS_SAVE_DIR = r'final_results'

os.makedirs(RESULTS_SAVE_DIR, exist_ok=True)

os.environ["HF_TOKEN"] = constants.HF_TOKEN

def read_json(file_path):
    with open(file_path) as json_file:
        return json.load(json_file)

human_pr_comments = read_json(HUMAN_PR_COMMENTS_PATH)
llm_pr_comments = read_json(LLM_PR_COMMENTS_PATH)

def get_embeddings(human_pr_comments, llm_pr_comments, use_cache=False, save_cache=False):
    if use_cache is True and os.path.exists(CACHE_FILE_NAME):
        with open(CACHE_FILE_NAME, 'rb') as file:
            embeddings_dict = pickle.load(file)
            return embeddings_dict

    model = SentenceTransformer("google/embeddinggemma-300m", model_kwargs={"torch_dtype": "bfloat16"})

    pr_url_array, language_array, human_review_comments_concat, llm_review_comments_concat = [], [], [], []

    for pr_url, human_reviews in tqdm(human_pr_comments.items()):
        if pr_url not in llm_pr_comments:
            continue

        human_review_comments = [human_review['body'] for human_review in human_reviews]
        human_review_comments = [clean_comment(human_review_comment) for human_review_comment in human_review_comments]

        llm_review = llm_pr_comments[pr_url]
        llm_review_comments = llm_review['parsed_comments']
        llm_review_comments = [clean_comment(llm_review_comment) for llm_review_comment in llm_review_comments]

        pr_url_array.append(pr_url)
        language_array.append(llm_review['language'])
        human_review_comments_concat.append('\n'.join(human_review_comments))
        llm_review_comments_concat.append('\n'.join(llm_review_comments))

    pr_url_array = np.array(pr_url_array)
    language_array = np.array(language_array)
    human_review_comments_concat = np.array(human_review_comments_concat)
    llm_review_comments_concat = np.array(llm_review_comments_concat)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    human_review_embeddings = model.encode_document(human_review_comments_concat, device=device, batch_size=128, show_progress_bar=True)
    llm_review_embeddings = model.encode_document(llm_review_comments_concat, device=device, batch_size=128, show_progress_bar=True)

    embeddings_dict = {
        'pr_url_array': pr_url_array,
        'language_array': language_array,
        'human_review_embeddings': human_review_embeddings,
        'llm_review_embeddings': llm_review_embeddings
    }

    if save_cache:
        with open(CACHE_FILE_NAME, 'wb') as file:
            pickle.dump(embeddings_dict, file)

    return embeddings_dict


embeddings_dict = get_embeddings(human_pr_comments, llm_pr_comments)

pr_url_array, language_array = embeddings_dict['pr_url_array'], embeddings_dict['language_array']
human_review_embeddings, llm_review_embeddings = embeddings_dict['human_review_embeddings'], embeddings_dict['llm_review_embeddings']

language_similarities_dict = dict()

for language in np.unique(language_array):
    language_mask = language_array == language
    language_filtered_pr_urls = pr_url_array[language_mask]

    language_cosine_similarities = []

    for pr_url in language_filtered_pr_urls:
        pr_url_mask = pr_url_array == pr_url
        pr_url_filtered_human_review_embeddings = human_review_embeddings[pr_url_mask]
        pr_url_filtered_llm_review_embeddings = llm_review_embeddings[pr_url_mask]

        language_cosine_similarities.extend(cosine_similarity(pr_url_filtered_human_review_embeddings, pr_url_filtered_llm_review_embeddings).flatten().tolist())

    print(f'{language} => Mean: {np.mean(language_cosine_similarities)} - Median: {np.median(language_cosine_similarities)} - Min: {np.min(language_cosine_similarities)} - Max: {np.max(language_cosine_similarities)}')
    language_similarities_dict[language] = language_cosine_similarities

plt.figure(dpi=300)
sns.boxplot(data=language_similarities_dict)
plt.title('Human vs. LLM Review Similarity (Concatenation Matching)')
plt.xlabel('Language')
plt.ylabel('Similarity Score')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_SAVE_DIR, 'Concatenation_similarities_boxplot.png'))
plt.close()

stats_data = []
for lang, scores in language_similarities_dict.items():
    stats_data.append({
        'Language': lang,
        'Min': np.min(scores),
        'Mean': np.mean(scores),
        'Max': np.max(scores)
    })

df_stats = pd.DataFrame(stats_data)

df_stats.to_html(os.path.join(RESULTS_SAVE_DIR, f'Concatenation_similarity_values.html'), index=False, float_format="{:.3f}".format)

# "Melt" the data to a long format for Seaborn
df_melted = df_stats.melt(id_vars='Language', var_name='Metric', value_name='Value')

plt.figure(figsize=(12, 6), dpi=300)
sns.barplot(data=df_melted, x='Language', y='Value', hue='Metric')

plt.title(f'Summary Statistics: Human vs. LLM Review Similarity (Concatenation Matching)')
plt.ylabel('Similarity Score')
plt.legend()  # title='Statistic', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_SAVE_DIR, f'Concatenation_similarities_barplot.png'))
plt.close()

