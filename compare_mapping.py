import json
import re
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def normalize_text(text):
    """Loại bỏ số thứ tự, ký hiệu a), b) ở đầu và khoảng trắng thừa"""
    if not text:
        return ""
    # loại bỏ đầu dòng kiểu "1. ", "2) ", "a) ", "b. "
    text = re.sub(r'^\s*(?:\d+\.|[a-zA-Z]\)|[a-zA-Z]\.)\s*', '', text.strip())
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def get_change_details(old_text, new_text):
    """
    Trả về dict chứa before_change, after_change, change_type
    """
    old_clean = normalize_text(old_text)
    new_clean = normalize_text(new_text)

    if not old_clean and new_clean:
        return {
            "before_change": "",
            "after_change": new_clean,
            "change_type": "added"
        }
    elif old_clean and not new_clean:
        return {
            "before_change": old_clean,
            "after_change": "",
            "change_type": "deleted"
        }
    elif old_clean != new_clean:
        # Nếu khác nhau
        # Lấy đoạn text khác nhau bằng difflib
        diff = difflib.SequenceMatcher(None, old_clean, new_clean)
        changed_parts = []
        for tag, i1, i2, j1, j2 in diff.get_opcodes():
            if tag in ("replace", "insert", "delete"):
                part = new_clean[j1:j2].strip()
                if part:
                    changed_parts.append(part)
        after_change = " ".join(changed_parts).strip()
        return {
            "before_change": old_clean,
            "after_change": after_change,
            "change_type": "modified" if after_change else "unchanged"
        }
    else:
        # Không có thay đổi
        return {
            "before_change": old_clean,
            "after_change": old_clean,
            "change_type": "unchanged"
        }


def extract_all_text_units(article):
    """Tạo danh sách các đoạn văn bản (Điều, Khoản, Điểm)"""
    units = []
    article_num = article.get("article_number", None)

    # Điều
    if article.get("full_text"):
        units.append({
            "id": article["article_id"],
            "article_number": article_num,
            "clause": None,
            "text": article["full_text"]
        })

    # Khoản
    for clause in article.get("clauses", []):
        if clause.get("full_text"):
            units.append({
                "id": clause.get("clause_id", article["article_id"]),
                "article_number": article_num,
                "clause": clause.get("clause", None),
                "text": clause["full_text"]
            })
        # Điểm
        for point in clause.get("points", []):
            if point.get("full_text"):
                units.append({
                    "id": point.get("point_id", clause.get("clause_id", article["article_id"])),
                    "article_number": article_num,
                    "clause": clause.get("clause", None),
                    "text": point["full_text"]
                })
    return units


def generate_mapping(file_2013, file_2024, output_file, threshold=0.6):
    # Load dữ liệu
    with open(file_2013, "r", encoding="utf-8") as f:
        art_2013 = json.load(f)
    with open(file_2024, "r", encoding="utf-8") as f:
        art_2024 = json.load(f)

    # Tạo danh sách đoạn văn
    units_2013, units_2024 = [], []
    for a in art_2013:
        units_2013.extend(extract_all_text_units(a))
    for a in art_2024:
        units_2024.extend(extract_all_text_units(a))

    texts_2013 = [u["text"] for u in units_2013]
    texts_2024 = [u["text"] for u in units_2024]

    # TF-IDF + cosine similarity
    vectorizer = TfidfVectorizer().fit(texts_2013 + texts_2024)
    tfidf_2013 = vectorizer.transform(texts_2013)
    tfidf_2024 = vectorizer.transform(texts_2024)
    sim_matrix = cosine_similarity(tfidf_2013, tfidf_2024)

    mapping = []

    # So sánh từng đoạn
    for i, sims in enumerate(sim_matrix):
        best_idx = sims.argmax()
        best_sim = sims[best_idx]

        old_unit = units_2013[i]

        if best_sim >= threshold:
            new_unit = units_2024[best_idx]
            change_info = get_change_details(old_unit["text"], new_unit["text"])

            mapping.append({
                "article_2013": old_unit["article_number"],
                "clause_2013": old_unit["clause"],
                "article_2024": new_unit["article_number"],
                "clause_2024": new_unit["clause"],
                "similarity": round(float(best_sim), 2),
                "change_type": change_info["change_type"],
                "before_change": change_info["before_change"],
                "after_change": change_info["after_change"]
            })
        else:
            change_info = get_change_details(old_unit["text"], "")
            mapping.append({
                "article_2013": old_unit["article_number"],
                "clause_2013": old_unit["clause"],
                "article_2024": None,
                "clause_2024": None,
                "similarity": 0,
                "change_type": change_info["change_type"],
                "before_change": change_info["before_change"],
                "after_change": change_info["after_change"]
            })

    # Lưu file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)

    print(f"✅ Saved {len(mapping)} mappings to {output_file}")


if __name__ == "__main__":
    generate_mapping(
        "output/articles_2013.json",
        "output/articles_2024.json",
        "output/mapping.json",
        threshold=0.6
    )
