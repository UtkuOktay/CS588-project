# CS588 Project
**Authors:** Canberk Aslan, Mert Barkın Er, Sinan Çavdar, Utku Oktay

## Prerequisites
### Python
You need to have Python installed. The dependencies can be installed by running the following command.

`pip install -r requirements.txt`

### LLMs
1. For comment cleaning and LLM-as-a-Judge evaluation, our project uses Gemma3:4b [1] model through Ollama API. Download Ollama: https://ollama.com/download

2. Once you install Ollama, you can download Gemma3:4b using the following command.<br />
`ollama pull gemma3:4b`

3. Then, start Ollama.<br />
`ollama serve`

### Tokens & Keys
You need to provide tokens for GitHub, OpenAI API and HuggingFace by filling them in constants.py.
```python
GITHUB_TOKEN = r"<GITHUB_TOKEN>"
OPENAI_API_KEY = r"<OPENAI_API_KEY>"
HF_TOKEN = r"<HUGGINGFACE_TOKEN>"
```

## How to Run
Since the pipeline involves multiple preprocessing, inference, and evaluation steps, running the entire workflow at once is time-consuming. Hence, we split it into separate scripts that produce intermediate results, so that subsequent scripts can use these outputs without rerunning previous steps. Consequently, you need to run the script one by one in the following order:

1. **`1_fetch_pr_diff_v5.py`**
2. **`2_merge_and_filter_prs_with_diffs.py`**
3. **`3_llm_inference.py`**
4. **`4_group_and_filter_comments_based_on_selected_prs.py`**
5. **`5_classify_meaningful_comments.py`**
6. **`6_merge_meaningful_pr_comments.py`**
7. **`7_parse_llm_responses.py`**
8. **`8_find_concat_similarities.py`**
9. **`9_find_pairwise_similarities.py`**
10. **`10_llm_as_judge.py`**
11. **`11_evaluate_llm_as_judge_responses.py`**

Once all scripts have run successfully, the results (charts and HTML files containing tables) will be saved in the **`final_results`** folder.

## References
[1] G. Team et al., ‘Gemma 3 Technical Report’, arXiv [cs.CL]. 2025.