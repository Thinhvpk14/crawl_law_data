import requests
from bs4 import BeautifulSoup
import json
import re

def clean_text(text):
    """Loại bỏ \r\n và khoảng trắng thừa"""
    return text.replace('\r', ' ').replace('\n', ' ').strip()

# ===== URL và request =====
url = "https://thuvienphapluat.vn/van-ban/Bat-dong-san/Luat-Dat-dai-2024-31-2024-QH15-523642.aspx"
headers = {"User-Agent": "Mozilla/5.0"}

response = requests.get(url, headers=headers)
response.encoding = "utf-8"
soup = BeautifulSoup(response.text, "html.parser")

# ===== Chỉ lấy nội dung trong div.content1 =====
content_div = soup.find("div", class_="content1")
if not content_div:
    print("Không tìm thấy div.content1")
    exit()

all_p = content_div.find_all("p")

articles = []
current_article = None
current_clause = None  # khoản hiện tại

for i, p in enumerate(all_p):
    text = clean_text(p.get_text())

    # ====== ĐIỀU ======
    a_dieu = p.find("a", attrs={"name": lambda x: x and x.startswith("dieu_")})
    if a_dieu or re.match(r"^Điều\s+\d+", text):
        if a_dieu:
            article_number = a_dieu["name"].replace("dieu_", "")
        else:
            # Tìm số điều từ text: "Điều 5." → 5
            match = re.match(r"^Điều\s+(\d+)", text)
            article_number = match.group(1) if match else "?"

        title = text
        current_article = {
            "article_id": f"L2024_#{article_number}",
            "law_version": "2024",
            "article_number": article_number,
            "title": title,
            "full_text": "",
            "clauses": []
        }
        articles.append(current_article)
        current_clause = None

        # Nếu p kế tiếp không phải khoản, thì lấy làm full_text
        if i + 1 < len(all_p):
            next_text = clean_text(all_p[i + 1].get_text())
            if not re.match(r"^\d+\.", next_text):  # không bắt đầu bằng "1.", "2.", ...
                current_article["full_text"] = next_text
        continue

    # ====== KHOẢN ======
    a_khoan = p.find("a", attrs={"name": lambda x: x and x.startswith("khoan_")})
    if a_khoan or re.match(r"^\d+\.", text):
        if a_khoan:
            parts = a_khoan["name"].split("_")
            if len(parts) >= 3:
                clause_num = parts[1]
            else:
                clause_num = "?"
        else:
            # Lấy số khoản từ đầu dòng: "1. ..." → "1"
            match = re.match(r"^(\d+)\.", text)
            clause_num = match.group(1) if match else "?"

        full_text = text
        current_clause = {
            "clause": clause_num,
            "full_text": full_text,
            "points": []
        }
        if current_article:
            current_article["clauses"].append(current_clause)
        continue

    # ====== ĐIỂM ======
    a_diem = p.find("a", attrs={"name": lambda x: x and x.startswith("diem_")})
    if a_diem or re.match(r"^[a-zA-Z]\)", text):
        if a_diem:
            parts = a_diem["name"].split("_")
            if len(parts) >= 4:
                point_char = parts[1]
            else:
                point_char = "?"
        else:
            # Lấy ký tự điểm: "a)" → "a"
            match = re.match(r"^([a-zA-Z])\)", text)
            point_char = match.group(1) if match else "?"

        if current_article and current_clause:
            current_clause["points"].append({
                "point": point_char,
                "full_text": text
            })
        continue

# ===== Lưu JSON =====
with open("output/articles_2024.json", "w", encoding="utf-8") as f:
    json.dump(articles, f, ensure_ascii=False, indent=4)

print(f"✅ Đã lưu {len(articles)} điều vào file articles_2024.json")
