import json
import os
import pickle

import pandas as pd
import torch
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from scipy.optimize import linear_sum_assignment
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

import constants
from comment_cleaning import clean_comment

HUMAN_PR_COMMENTS_PATH = r'meaningful_pr_comments_merged.json'
LLM_PR_COMMENTS_PATH = r'parsed_llm_responses.json'

CACHE_FILE_NAME = r'pairwise_embeddings.pkl'

RESULTS_SAVE_DIR = r'final_results'

os.makedirs(RESULTS_SAVE_DIR, exist_ok=True)

os.environ["HF_TOKEN"] = constants.HF_TOKEN

def read_json(file_path):
    with open(file_path) as json_file:
        return json.load(json_file)

human_pr_reviews = read_json(HUMAN_PR_COMMENTS_PATH)
llm_pr_reviews = read_json(LLM_PR_COMMENTS_PATH)

# ---------------------------------------------------------
# HUNGARIAN MATCHING (Optimal Assignment)
# ---------------------------------------------------------
def hungarian_match(sim):
    # Hungarian minimizes cost → convert similarity → cost
    cost = -sim  # maximize similarity = minimize negative similarity

    # Solve optimal assignment
    row_ind, col_ind = linear_sum_assignment(cost)

    matched_sims = sim[row_ind, col_ind]
    return np.mean(matched_sims)

def get_embeddings(human_pr_reviews, llm_pr_reviews, use_cache=False, save_cache=False):
    if use_cache is True and os.path.exists(CACHE_FILE_NAME):
        with open(CACHE_FILE_NAME, 'rb') as file:
            embeddings_dict = pickle.load(file)
            return embeddings_dict

    model = SentenceTransformer("google/embeddinggemma-300m", model_kwargs={"torch_dtype": "bfloat16"})

    human_pr_url_array, all_human_review_comments = [], []
    llm_pr_url_array, all_llm_review_comments = [], []

    for pr_url, human_reviews in tqdm(human_pr_reviews.items()):
        if pr_url not in llm_pr_reviews:
            continue

        human_review_comments = [human_review['body'] for human_review in human_reviews]
        human_review_comments = [clean_comment(human_review_comment) for human_review_comment in human_review_comments]

        llm_review = llm_pr_reviews[pr_url]
        llm_review_comments = llm_review['parsed_comments']
        llm_review_comments = [clean_comment(llm_review_comment) for llm_review_comment in llm_review_comments]

        for human_review_comment in human_review_comments:
            human_pr_url_array.append(pr_url)
            all_human_review_comments.append(human_review_comment)

        for llm_review_comment in llm_review_comments:
            llm_pr_url_array.append(pr_url)
            all_llm_review_comments.append(llm_review_comment)

    human_pr_url_array, all_human_review_comments = np.array(human_pr_url_array), np.array(all_human_review_comments)
    llm_pr_url_array, all_llm_review_comments = np.array(llm_pr_url_array), np.array(all_llm_review_comments)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    human_review_embeddings = model.encode_document(all_human_review_comments, device=device, batch_size=128, show_progress_bar=True)
    llm_review_embeddings = model.encode_document(all_llm_review_comments, device=device, batch_size=128, show_progress_bar=True)

    embeddings_dict = {
        'human_pr_url_array': human_pr_url_array,
        'human_review_embeddings': human_review_embeddings,

        'llm_pr_url_array': llm_pr_url_array,
        'llm_review_embeddings': llm_review_embeddings
    }

    if save_cache:
        with open(CACHE_FILE_NAME, 'wb') as file:
            pickle.dump(embeddings_dict, file)

    return embeddings_dict


def greedy_match(similarity_matrix):
    """
    Computes average similarity by transposing the input matrix to
    ensure GPT comments are rows and Human comments are columns.
    """
    # Transpose to ensure Rows = GPT, Cols = Human
    # This aligns with the 'for each GPT, find best Human' logic
    matrix = similarity_matrix.T

    n_gpt, n_human = matrix.shape

    if n_gpt == 0 or n_human == 0:
        return 0.0

    total_similarity = 0.0
    used_human_indices = set()
    match_count = 0

    for g_idx in range(n_gpt):
        if len(used_human_indices) >= n_human:
            break

        # We work on a copy of the row to mask used indices
        row_sims = matrix[g_idx].copy()

        if used_human_indices:
            row_sims[list(used_human_indices)] = -np.inf

        best_h_idx = np.argmax(row_sims)
        best_sim = row_sims[best_h_idx]

        if best_sim != -np.inf:
            total_similarity += best_sim
            used_human_indices.add(best_h_idx)
            match_count += 1

    return total_similarity / match_count if match_count > 0 else 0.0

def find_similarities(matching_strategy_functions, matching_strategy_functions_names):
    embeddings_dict = get_embeddings(human_pr_reviews, llm_pr_reviews)

    human_pr_url_array, human_review_embeddings = embeddings_dict['human_pr_url_array'], embeddings_dict['human_review_embeddings']
    llm_pr_url_array, llm_review_embeddings = embeddings_dict['llm_pr_url_array'], embeddings_dict['llm_review_embeddings']

    language_prs_dict = dict()

    for pr_url, llm_pr_review in llm_pr_reviews.items():
        language = llm_pr_review['language']

        prs_in_lang = language_prs_dict.get(language, [])
        prs_in_lang.append(pr_url)
        language_prs_dict[language] = prs_in_lang


    similarities_df_rows = []

    for language, pr_urls in language_prs_dict.items():
        language_cosine_similarities = []

        for pr_url in tqdm(pr_urls):
            pr_url_filtered_human_review_embeddings = human_review_embeddings[human_pr_url_array == pr_url]
            pr_url_filtered_llm_review_embeddings = llm_review_embeddings[llm_pr_url_array == pr_url]

            if pr_url_filtered_human_review_embeddings.shape[0] == 0:
                continue

            pr_cosine_similarities = cosine_similarity(pr_url_filtered_human_review_embeddings, pr_url_filtered_llm_review_embeddings)

            similarities_df_row = [pr_url, language]

            for matching_strategy_function in matching_strategy_functions:
                pr_reduced_similarity = matching_strategy_function(pr_cosine_similarities)
                similarities_df_row.append(pr_reduced_similarity)

            similarities_df_rows.append(similarities_df_row)

    similarities_df = pd.DataFrame(similarities_df_rows, columns=['pr_url', 'language'] + matching_strategy_functions_names)
    stats_dfs = []

    for matching_strategy_name in matching_strategy_functions_names:
        language_similarities_dict = dict()

        for language in similarities_df['language'].unique():
            language_filtered_df = similarities_df[similarities_df['language'] == language]
            language_similarities_dict[language] = language_filtered_df[matching_strategy_name]

        plt.figure(dpi=300)
        sns.boxplot(data=language_similarities_dict)
        plt.title(f'Human vs. LLM Review Similarity ({matching_strategy_name} Matching)')
        plt.xlabel('Language')
        plt.ylabel('Similarity Score')
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_SAVE_DIR, f'{matching_strategy_name}_similarities_boxplot.png'))
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

        df_stats.to_html(os.path.join(RESULTS_SAVE_DIR, f'{matching_strategy_name}_similarity_values.html'), index=False, float_format="{:.3f}".format)

        stats_dfs.append(df_stats[['Language', 'Mean']].rename(columns={'Mean': matching_strategy_name}))

        # "Melt" the data to a long format for Seaborn
        df_melted = df_stats.melt(id_vars='Language', var_name='Metric', value_name='Value')

        plt.figure(figsize=(12, 6), dpi=300)
        sns.barplot(data=df_melted, x='Language', y='Value', hue='Metric')

        plt.title(f'Summary Statistics: Human vs. LLM Review Similarity ({matching_strategy_name} Matching)')
        plt.ylabel('Similarity Score')
        plt.legend()  # title='Statistic', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_SAVE_DIR, f'{matching_strategy_name}_similarities_barplot.png'))
        plt.close()

    final_stats_df = stats_dfs[0]
    for i in range(1, len(stats_dfs)):
        final_stats_df = final_stats_df.merge(stats_dfs[i], on='Language')

    final_stats_df.to_html(os.path.join(RESULTS_SAVE_DIR, f'all_matching_strategies_similarity_values.html'), index=False, float_format="{:.3f}".format)

find_similarities([greedy_match, hungarian_match, np.max], ['Greedy', 'Hungarian', 'Top-1'])