import re

term = "1. Trường hợp thu hồi đất"
term = re.sub(r'^\s*(\d+\.\s*|[a-z]\)\s*)', '', term, flags=re.IGNORECASE)
print(term)
