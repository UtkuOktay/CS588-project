import ast
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import constants

# 1. PATHS
INPUT_DIR = r"dataset_raw_json_files"
OUTPUT_DIR = r"pr_diffs"
CACHE_DIR = ".gh_cache"

# 2. LIMITS
TARGET_PR_COUNT = 20000  # Number of PRs to download
MAX_WORKERS = 1         # Parallel downloads (set to 5-10)
# =============================================================

PR_URL_RE = re.compile(
    r"https?://(?:www\.)?github\.com/(?P<owner>[^/]+)/(?P<repo>[^/]+)/pull/(?P<number>\d+)",
    re.IGNORECASE,
)

class GitHubClient:
    def __init__(self, token: str):
        self.token = token.strip()
        self.session = self._make_session()
        self.api_url = "https://api.github.com"
        self.gql_url = "https://api.github.com/graphql"

    def _make_session(self) -> requests.Session:
        sess = requests.Session()
        retry = Retry(
            total=5, backoff_factor=1, status_forcelist=[500, 502, 503, 504],
            allowed_methods=["GET", "POST"]
        )
        adapter = HTTPAdapter(max_retries=retry)
        sess.mount("https://", adapter)
        
        headers = {
            "User-Agent": "Research-Script/1.0",
            "Accept": "application/vnd.github.v3+json",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        sess.headers.update(headers)
        return sess

    def get_stars_batch(self, repos: List[str]) -> Dict[str, int]:
        """Fetches stars for up to 50 repos at once."""
        if not self.token:
            print("Warning: No token. GraphQL disabled.")
            return {}

        query_parts = []
        alias_map = {}
        
        for i, repo_full in enumerate(repos):
            if "/" not in repo_full: continue
            owner, name = repo_full.split("/", 1)
            # GraphQL aliases cannot contain special chars, so we use r0, r1...
            alias = f"r{i}"
            alias_map[alias] = repo_full
            query_parts.append(
                f'{alias}: repository(owner: "{owner}", name: "{name}") {{ stargazerCount }}'
            )

        if not query_parts:
            return {}

        query = "query { " + " ".join(query_parts) + " }"
        
        try:
            resp = self.session.post(self.gql_url, json={"query": query})
            if resp.status_code != 200:
                return {}
            
            data = resp.json().get("data", {})
            results = {}
            if data:
                for alias, repo_data in data.items():
                    if repo_data:
                        full_name = alias_map.get(alias)
                        if full_name:
                            results[full_name] = repo_data.get("stargazerCount", 0)
            return results
        except Exception:
            return {}

    def get_pr_details(self, repo: str, number: int) -> Any:
        start_time = time.time()
        
        url = f"{self.api_url}/repos/{repo}/pulls/{number}"
        try:
            # 1. Metadata (Title, Body/Description)
            resp_meta = self.session.get(url, timeout=10)
            if resp_meta.status_code in [403, 429]: return "RATE_LIMIT"
            if resp_meta.status_code != 200: return None
            data = resp_meta.json()
            
            # 2. Diff
            resp_diff = self.session.get(
                url, headers={"Accept": "application/vnd.github.v3.diff"}, timeout=15
            )
            diff_text = resp_diff.text if resp_diff.status_code == 200 else ""

            elapsed_time = time.time() - start_time
            time_to_sleep = 1.75 - elapsed_time
            time_to_sleep = 0 if time_to_sleep < 0 else time_to_sleep
            time.sleep(time_to_sleep)
            
            return {
                "repo": repo,
                "pr_number": number,
                "html_url": data.get("html_url"),
                "title": data.get("title"),
                "description": data.get("body"),
                "diff": diff_text,
                "created_at": data.get("created_at"),
                "state": data.get("state")
            }
        except Exception:
            return None

def load_python_style_dataset(path: str) -> Dict[str, Any]:
    print(f"Loading dataset from: {path} ...")
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")
        
    raw_text = p.read_text(encoding="utf-8", errors="replace").strip()
    
    # Strip variable assignment if present (e.g. "data = {...}")
    if "=" in raw_text[:50] and not raw_text.startswith("{"):
        _, raw_text = raw_text.split("=", 1)
        raw_text = raw_text.strip()

    # 1. Try standard JSON
    try:
        return json.loads(raw_text)
    except:
        pass

    # 2. Try Robust Python Eval (Handles 'nan', 'null', etc.)
    safe_context = {
        "null": None, "true": True, "false": False,
        "nan": None, "inf": float('inf'),
        "None": None, "True": True, "False": False
    }
    
    try:
        return eval(raw_text, {"__builtins__": {}}, safe_context)
    except Exception as e:
        print(f"Failed to parse dataset. Error: {e}")
        raise e

def extract_prs_from_dataset(dataset: Dict[str, Any]) -> Dict[str, Set[int]]:
    repo_prs = {}
    for repo_name, items in dataset.items():
        if not isinstance(items, list): continue
        unique_nums = set()
        for item in items:
            if not isinstance(item, dict): continue
            url = item.get("html_url", "")
            match = PR_URL_RE.search(str(url))
            if match:
                unique_nums.add(int(match.group('number')))
        if unique_nums:
            repo_prs[repo_name] = unique_nums
    return repo_prs

# --- STAR CACHING FUNCTIONS ---
def load_star_cache(cache_path: Path) -> Dict[str, int]:
    if not cache_path.exists():
        return {}
    try:
        return json.loads(cache_path.read_text(encoding="utf-8"))
    except:
        return {}

def save_star_cache(cache_path: Path, data: Dict[str, int]):
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

def main(input_path, output_path):
    if "YOUR_TOKEN" in constants.GITHUB_TOKEN or not constants.GITHUB_TOKEN:
        print("!!! ERROR: Please paste your GitHub Token in the configuration section !!!")
        return
        
    client = GitHubClient(constants.GITHUB_TOKEN)
    
    # 1. Load Dataset
    try:
        data = load_python_style_dataset(input_path)
    except Exception:
        return

    # 2. Parse Repos
    repo_to_prs = extract_prs_from_dataset(data)
    all_repos = list(repo_to_prs.keys())
    print(f"Found {len(all_repos)} unique repositories in dataset.")

    # 3. Fetch Stars (With Caching)
    cache_file = Path(CACHE_DIR) / f"{Path(input_path).stem}_repo_stars_cache.json"
    repo_stars = load_star_cache(cache_file)
    print(f"Loaded {len(repo_stars)} repositories from star cache.")
    
    repos_to_fetch = [r for r in all_repos if r not in repo_stars]
    
    if repos_to_fetch:
        print(f"Fetching stars for {len(repos_to_fetch)} missing repositories...")
        batch_size = 40
        for i in range(0, len(repos_to_fetch), batch_size):
            batch = repos_to_fetch[i : i + batch_size]
            stars = client.get_stars_batch(batch)
            repo_stars.update(stars)
            
            # Save cache periodically (every 200 repos) to avoid data loss
            if i % 200 == 0:
                save_star_cache(cache_file, repo_stars)
                print(f"  Checked {i}/{len(repos_to_fetch)}...", end="\r")
        
        # Final save
        save_star_cache(cache_file, repo_stars)
        print("\nStar fetching complete.")
    else:
        print("All repository stars are already cached!")

    # 4. Sort and Filter (Highest Stars -> Lowest)
    print("Sorting repositories by stars (Descending)...")
    sorted_repos = sorted(all_repos, key=lambda r: repo_stars.get(r, 0), reverse=True)
    
    selected_tasks = []
    for repo in sorted_repos:
        prs = sorted(list(repo_to_prs[repo]))
        for pr_num in prs:
            if len(selected_tasks) >= TARGET_PR_COUNT: break
            selected_tasks.append((repo, pr_num))
        if len(selected_tasks) >= TARGET_PR_COUNT: break

    print(f"Targeting top {len(selected_tasks)} PRs from highest star repositories.")

    # 5. Resume Capability (Don't re-download PRs we already have)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    finished_keys = set()
    if out_path.exists():
        with out_path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    finished_keys.add(f"{obj['repo']}#{obj['pr_number']}")
                except: pass
    
    todo_tasks = [t for t in selected_tasks if f"{t[0]}#{t[1]}" not in finished_keys]
    print(f"Already done: {len(finished_keys)}. Remaining: {len(todo_tasks)}")

    if not todo_tasks:
        print("All target PRs already fetched. Done.")
        return

    # 6. Execute Parallel Download
    print(f"Starting PR download with {MAX_WORKERS} workers...")
    count = 0
    
    # Ensure workers is at least 1
    workers = MAX_WORKERS if MAX_WORKERS > 0 else 5

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(client.get_pr_details, r, n): (r, n) for r, n in todo_tasks}
        
        with out_path.open("a", encoding="utf-8") as f_out:
            pr_fetching_start_time = time.time()

            for future in as_completed(futures):
                r, n = futures[future]
                try:
                    res = future.result()
                    if res == "RATE_LIMIT":
                        print("\n!!! RATE LIMIT REACHED !!!")
                        print("Stopping gracefully to protect your account.")
                        executor.shutdown(wait=False, cancel_futures=True)
                        break
                    
                    if res:
                        # Inject star count into the final JSON for reference
                        res["repo_stars"] = repo_stars.get(r, 0)
                        f_out.write(json.dumps(res) + "\n")
                        f_out.flush()
                        
                        count += 1
                        if count % 10 == 0:
                            elapsed_time = time.time() - pr_fetching_start_time
                            elapsed_time_str = f'{elapsed_time // 60} mins and {elapsed_time % 60} secs'
                            print(f"Fetched {count}/{len(todo_tasks)} PRs... Elapsed time: {elapsed_time_str}. {count / elapsed_time} pr/s", end="\r")
                            
                except Exception as e:
                    print(f"Err {r}#{n}: {e}")

    print(f"\nCompleted. Data saved to {output_path}")

if __name__ == "__main__":
    files = [file for file in os.listdir(INPUT_DIR) if file.endswith(".json")]
    for file in files:
        output_filename = f'{file.split("-")[-1].replace(".json", "").lower()}_prs.jsonl'
        main(os.path.join(INPUT_DIR, file), os.path.join(OUTPUT_DIR, output_filename))