import requests
from bs4 import BeautifulSoup
import json

def clean_text(text):
    """Loại bỏ \r\n và khoảng trắng thừa"""
    return text.replace('\r', ' ').replace('\n', ' ').strip()

# ===== URL và request =====
url = "https://thuvienphapluat.vn/van-ban/Bat-dong-san/Luat-dat-dai-2013-215836.aspx"
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
    # ===== ĐIỀU =====
    a_dieu = p.find("a", attrs={"name": lambda x: x and x.startswith("dieu_")})
    if a_dieu:
        article_number = a_dieu["name"].replace("dieu_", "")
        b_tag = a_dieu.find("b")
        title = clean_text(b_tag.get_text()) if b_tag else clean_text(p.get_text())

        # Tạo object Điều
        current_article = {
            "article_id": f"L2013_#{article_number}",
            "law_version": "2013",
            "article_number": article_number,
            "title": title,
            "full_text": "",  # ✅ thêm field full_text cho Điều
            "clauses": []     # chứa danh sách khoản
        }
        articles.append(current_article)
        current_clause = None

        # ✅ Kiểm tra p kế tiếp có khoản không
        if i + 1 < len(all_p):
            next_p = all_p[i + 1]
            has_clause = next_p.find("a", attrs={"name": lambda x: x and x.startswith("khoan_")})
            if not has_clause:
                # Nếu không có khoản, lấy nội dung kế tiếp làm full_text
                current_article["full_text"] = clean_text(next_p.get_text())
        continue

    # ===== KHOẢN =====
    a_khoan = p.find("a", attrs={"name": lambda x: x and x.startswith("khoan_")})
    if a_khoan and current_article:
        parts = a_khoan["name"].split("_")  # ví dụ: ["khoan", "3", "3"] hoặc ["khoan", "3", "3", "3"]

        # lấy số khoản và số điều linh hoạt
        if len(parts) >= 3:
            clause_num = parts[1]
            article_num = parts[-1]  # phần cuối cùng luôn là số điều
            full_text = clean_text(p.get_text())

            current_clause = {
                "clause": clause_num,
                "full_text": full_text,
                "points": []  # chứa các điểm a, b, c
            }
            current_article["clauses"].append(current_clause)
        continue

    # ===== ĐIỂM =====
    a_diem = p.find("a", attrs={"name": lambda x: x and x.startswith("diem_")})
    if a_diem and current_article and current_clause:
        parts = a_diem["name"].split("_")  # ["diem", x, y, z]
        if len(parts) == 4:
            point_char, clause_num, article_num = parts[1], parts[2], parts[3]
            full_text = clean_text(p.get_text())
            current_clause["points"].append({
                "point": point_char,
                "full_text": full_text
            })
        continue

# ===== Lưu JSON =====
with open("output/articles_2013.json", "w", encoding="utf-8") as f:
    json.dump(articles, f, ensure_ascii=False, indent=4)

print(f"✅ Đã lưu {len(articles)} điều vào file articles_2013.json")
