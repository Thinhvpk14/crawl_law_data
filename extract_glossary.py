import json
import re
from collections import defaultdict

# Regex: tìm cụm "term là definition" hoặc "term có nghĩa là definition"
TERM_RE = re.compile(
    r'(.+?)\s+(?:là|có nghĩa là)\s+(.*?)(?:[.;\n]|$)',
    flags=re.IGNORECASE | re.S
)

def clean_text(text):
    """Loại bỏ \r\n và khoảng trắng thừa"""
    return text.replace('\r', ' ').replace('\n', ' ').strip()

def clean_term(term):
    """Loại bỏ số thứ tự hoặc ký hiệu a), b) ở đầu term"""
    term = term.strip()
    # Loại bỏ 1. , 2. , a) , b) ở đầu
    term = re.sub(r'^\s*(\d+\.\s*|[a-z]\)\s*)', '', term, flags=re.IGNORECASE)
    return term

def process_articles_file(articles, term_dict):
    """
    Duyệt từng article để extract term/definition,
    cập nhật vào term_dict (term -> {definition, related_articles})
    """
    for article in articles:
        article_id = article.get("article_id")

        # Gom tất cả text để quét term
        text_list = []

        if article.get("full_text"):
            text_list.append(article["full_text"])
        
        for clause in article.get("clauses", []):
            if clause.get("full_text"):
                text_list.append(clause["full_text"])
            
            for point in clause.get("points", []):
                if point.get("full_text"):
                    text_list.append(point["full_text"])

        # Quét từng text để tìm term
        for text in text_list:
            text = clean_text(text)
            for match in TERM_RE.finditer(text):
                term, definition = match.groups()
                term = clean_term(term)
                definition = definition.strip()

                if term in term_dict:
                    term_dict[term]["related_articles"].add(article_id)
                else:
                    term_dict[term]["definition"] = definition
                    term_dict[term]["related_articles"].add(article_id)

def build_glossary(output_file, articles_files):
    """
    articles_files: list các file JSON articles
    """
    term_dict = defaultdict(lambda: {"definition": "", "related_articles": set()})

    for file in articles_files:
        with open(file, "r", encoding="utf-8") as f:
            articles = json.load(f)
        process_articles_file(articles, term_dict)

    # Chuyển sang list glossary_terms
    glossary_terms = []
    for term, info in sorted(term_dict.items(), key=lambda x: x[0].lower()):
        glossary_terms.append({
            "term": term,
            "definition": info["definition"],
            "related_articles": sorted(info["related_articles"])
        })

    # Lưu JSON
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(glossary_terms, f, ensure_ascii=False, indent=2)

    print(f"✅ Saved {len(glossary_terms)} glossary terms to {output_file}")

if __name__ == "__main__":
    # Ví dụ dùng hai file articles
    articles_files = ["output/articles_2013.json", "output/articles_2024.json"]
    build_glossary("output/glossary.json", articles_files)
