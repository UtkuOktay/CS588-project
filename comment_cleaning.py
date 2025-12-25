import re

url_regex = re.compile(r"https?://\S+")
md_regex = re.compile(r"[`*_>#~\-]{1,}")

def remove_urls(text):
    return url_regex.sub("", text)

def remove_markdown(text):
    return md_regex.sub(" ", text)

def clean_comment(text):
    if not isinstance(text, str):
        return ""
    text = remove_urls(text)
    text = remove_markdown(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
