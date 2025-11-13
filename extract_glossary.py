import json
import re
from collections import defaultdict

# Regex: tìm cụm "term là definition" hoặc "term có nghĩa là definition"
TERM_RE = re.compile(
    r'(.+?)\s+(?:là|có nghĩa là)\s+(.*?)(?:[.;\n]|$)',
    flags=re.IGNORECASE | re.S
)

# Danh sách từ khóa "động từ/ghi chú" thường xuất hiện trong câu giải thích chứ không phải thuật ngữ
VERB_STOPWORDS = {
    "được", "gồm", "bao gồm", "là", "lưu ý", "có", "thực hiện", "là việc", "gọi là",
    "là", "gồm:", "gồm các", "trường hợp", "là những", "là các"
}

def clean_text(text):
    """Loại bỏ \r\n và khoảng trắng thừa"""
    return text.replace('\r', ' ').replace('\n', ' ').strip()

def clean_term(term):
    """Loại bỏ số thứ tự hoặc ký hiệu a), b) ở đầu term"""
    term = term.strip()
    # Loại bỏ 1. , 2. , a) , b) ở đầu
    term = re.sub(r'^\s*(\d+\.\s*|[a-zA-Z]\)\s*)', '', term, flags=re.IGNORECASE)
    # Loại bỏ dấu hai chấm ở cuối nếu có (thường do format)
    term = term.rstrip(':').strip()
    return term

def is_likely_term(term,
                   max_words=8,
                   max_chars=120,
                   max_commas=1,
                   stopword_threshold=1):
    """
    Heuristic kiểm tra xem chuỗi phía trước 'là' có khả năng là thuật ngữ hay không.
    Trả về True nếu có vẻ như thuật ngữ; False nếu có vẻ là câu mô tả/đoạn văn.
    Tham số có thể điều chỉnh:
      - max_words: số từ tối đa chấp nhận (nhiều hơn dễ là câu)
      - max_chars: số ký tự tối đa
      - max_commas: số dấu phẩy tối đa (nhiều dấu phẩy => có khả năng là câu)
      - stopword_threshold: nếu chứa >= threshold stopwords thì bỏ qua
    """

    if not term:
        return False

    # Xử lý sơ bộ
    t = term.strip()

    # Nếu quá ngắn (1 ký tự) thì bỏ
    if len(t) < 2:
        return False

    # loại bỏ ngoặc, gạch đầu dòng dư
    t = re.sub(r'^[\-\–\—\•\u2022\s]+', '', t)
    t = t.strip()

    # Kiểm tra ký tự
    if len(t) > max_chars:
        return False

    # word count (dựa trên space) — tiếng Việt dùng space tốt
    words = re.split(r'\s+', t)
    word_count = len(words)
    if word_count > max_words:
        return False

    # quá nhiều dấu câu => có khả năng là câu mô tả
    comma_count = t.count(',')
    if comma_count > max_commas:
        return False

    # Nếu trong term xuất hiện động từ/stopwords nhiều => nhiều khả năng không phải thuật ngữ
    found_sw = 0
    lowered = t.lower()
    for sw in VERB_STOPWORDS:
        if sw in lowered:
            found_sw += 1
            if found_sw >= stopword_threshold:
                return False

    # Nếu term chứa đầy đủ một cụm động từ ở cuối như "là việc", "gồm" => bỏ
    if re.search(r'\b(là việc|gồm|bao gồm|là những|là các|trường hợp)\b', lowered):
        return False

    # Nếu term chứa dấu chấm (.) => có thể là sentence fragment
    if '.' in t and len(t) > 10:
        return False

    # Nếu term kết thúc bằng một giới từ/động từ thường xuất hiện ở phần mô tả, bỏ
    if re.search(r'\b(?:là|gồm|gọi|được|bao|chỉ)\s*$', lowered):
        return False

    # Nếu đi qua hết check thì coi như khả năng lớn là thuật ngữ
    return True

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
                term_candidate, definition = match.groups()
                term_candidate = clean_term(term_candidate)
                definition = definition.strip()

                # Bổ sung: chỉ chấp nhận khi is_likely_term == True
                if not is_likely_term(term_candidate):
                    # nếu muốn debug: uncomment dòng dưới để in ra những candidate bị bỏ
                    # print("SKIP candidate (not term-like):", repr(term_candidate))
                    continue

                term = term_candidate
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