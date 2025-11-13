#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
mapping_generator.py
So sánh hai bộ luật (articles_2013.json, articles_2024.json) và sinh mapping:
  - chỉ xuất changed entries (added/deleted/modified)
  - giữ article_id/article_number/clause_id/point_id
  - hỗ trợ Hungarian matching nếu scipy có sẵn, fallback greedy nếu không
  - detect split/merge nhưng theo yêu cầu: split -> mark deleted (old), merge -> mark added (new)
  - regex bắt cả ký tự tiếng Việt (ví dụ 'đ)')
"""

import json
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# cố gắng import Hungarian (linear_sum_assignment)
try:
    from scipy.optimize import linear_sum_assignment
    _HUNGARIAN_AVAILABLE = True
except Exception:
    _HUNGARIAN_AVAILABLE = False

# ----------------- CẤU HÌNH (có thể chỉnh) -----------------
TFIDF_NGRAM = (1, 2)
MATCH_THRESHOLD = 0.60        # ngưỡng để coi 1->1 match hợp lệ
SPLIT_UNIT_THRESH = 0.35      # candidate new units có sim >= đây được xét cho split
SPLIT_SUM_THRESH = 0.75       # tổng sim nhiều new >= đây => xem là split
MERGE_UNIT_THRESH = 0.35      # tương tự cho merge (many old -> one new)
MERGE_SUM_THRESH = 0.75
# ----------------------------------------------------------

# -- Normalize / regex --
# Regex mới bắt chữ Latin mở rộng (bao gồm chữ tiếng Việt như đ, ă, â, ê, ô, ơ, ư, ...)
_LEADING_MARK_RE = re.compile(
    r'^\s*(?:\d+\.\s*|\d+\)\s*|[A-Za-zÀ-ỹ]\)\s*|[A-Za-zÀ-ỹ]\.\s*)',
    flags=re.IGNORECASE | re.UNICODE
)

def normalize_for_compare(text):
    """
    Chuẩn hoá để tính similarity:
      - loại bỏ tiền tố mục (1., 1), a), đ), a., ...)
      - thay newline/tab bằng space
      - rút gọn khoảng trắng
    """
    if not text:
        return ""
    t = text.strip()
    t = _LEADING_MARK_RE.sub('', t)
    t = t.replace('\r', ' ').replace('\n', ' ')
    t = re.sub(r'\s+', ' ', t)
    return t.strip()

def normalize_for_output(text):
    """
    Chuẩn hoá khi lưu before_change/after_change:
      - loại bỏ tiền tố mục đầu dòng (1., a), đ) ...)
      - giữ nguyên phần còn lại (không cắt diff)
    """
    if not text:
        return ""
    t = text.strip()
    t = _LEADING_MARK_RE.sub('', t)
    t = t.replace('\r', ' ').replace('\n', ' ')
    t = re.sub(r'\s+', ' ', t)
    return t.strip()

# -- Extract units from article JSON --
def extract_units(article):
    """
    Trả về list đơn vị (article / clause / point) với các field:
      { article_id, article_number, clause_id, point_id, text }
    Lưu ý: clause_id lấy ưu tiên clause['clause_id'] hoặc clause['clause'] nếu có.
    point_id lấy ưu tiên point['point_id'] hoặc point['point'].
    """
    units = []
    article_id = article.get("article_id")
    article_number = article.get("article_number")

    # Article-level full_text
    if article.get("full_text"):
        units.append({
            "article_id": article_id,
            "article_number": article_number,
            "clause_id": None,
            "point_id": None,
            "text": article["full_text"]
        })

    # Clauses
    for clause in article.get("clauses", []):
        clause_id = clause.get("clause_id") or clause.get("clause")
        if clause.get("full_text"):
            units.append({
                "article_id": article_id,
                "article_number": article_number,
                "clause_id": clause_id,
                "point_id": None,
                "text": clause["full_text"]
            })
        # Points inside clause
        for point in clause.get("points", []):
            point_id = point.get("point_id") or point.get("point")
            if point.get("full_text"):
                units.append({
                    "article_id": article_id,
                    "article_number": article_number,
                    "clause_id": clause_id,
                    "point_id": point_id,
                    "text": point["full_text"]
                })
    return units

# -- Matching helpers --
def global_optimal_matching(sim_matrix, use_hungarian=True):
    """
    Trả về danh sách cặp (i, j, sim).
    Nếu có Hungarian (scipy) và use_hungarian True -> áp dụng Hungarian trên ma trận vuông.
    Ngược lại -> greedy one-to-one (sắp theo sim giảm dần, gán nếu cả hai chưa gán).
    """
    n_old, n_new = sim_matrix.shape
    pairs = []
    if use_hungarian and _HUNGARIAN_AVAILABLE:
        M = max(n_old, n_new)
        cost = np.ones((M, M), dtype=float)
        cost[:n_old, :n_new] = 1.0 - sim_matrix  # cost = 1 - sim
        row_ind, col_ind = linear_sum_assignment(cost)
        for r, c in zip(row_ind, col_ind):
            if r < n_old and c < n_new:
                pairs.append((int(r), int(c), float(sim_matrix[r, c])))
        return pairs
    else:
        # Greedy one-to-one
        idxs = np.transpose(np.nonzero(sim_matrix >= 0.0))
        cand = []
        for r, c in idxs:
            cand.append((int(r), int(c), float(sim_matrix[r, c])))
        cand.sort(key=lambda x: x[2], reverse=True)
        matched_old = set(); matched_new = set()
        for r, c, s in cand:
            if r in matched_old or c in matched_new:
                continue
            matched_old.add(r); matched_new.add(c)
            pairs.append((r, c, s))
        return pairs

def detect_splits_and_merges(sim_matrix):
    """
    Dò các split (old -> nhiều new) và merge (nhiều old -> new) theo thresholds cấu hình.
    Trả về (splits, merges) với:
      splits: { old_index: [(new_index, sim), ...], ... }
      merges: { new_index: [(old_index, sim), ...], ... }
    """
    n_old, n_new = sim_matrix.shape
    splits = {}
    for i in range(n_old):
        sims = [(j, float(sim_matrix[i, j])) for j in range(n_new) if float(sim_matrix[i, j]) >= SPLIT_UNIT_THRESH]
        sims.sort(key=lambda x: x[1], reverse=True)
        if len(sims) > 1:
            total = sum(s for (_, s) in sims)
            if total >= SPLIT_SUM_THRESH:
                splits[i] = sims
    merges = {}
    for j in range(n_new):
        sims = [(i, float(sim_matrix[i, j])) for i in range(n_old) if float(sim_matrix[i, j]) >= MERGE_UNIT_THRESH]
        sims.sort(key=lambda x: x[1], reverse=True)
        if len(sims) > 1:
            total = sum(s for (_, s) in sims)
            if total >= MERGE_SUM_THRESH:
                merges[j] = sims
    return splits, merges

# -- Main generator --
def generate_mapping(file_2013, file_2024, output_file,
                     match_threshold=MATCH_THRESHOLD,
                     use_hungarian=True):
    # Load JSON files
    with open(file_2013, "r", encoding="utf-8") as f:
        art_2013 = json.load(f)
    with open(file_2024, "r", encoding="utf-8") as f:
        art_2024 = json.load(f)

    # Extract units
    units_old = []
    units_new = []
    for a in art_2013:
        units_old.extend(extract_units(a))
    for a in art_2024:
        units_new.extend(extract_units(a))

    texts_old = [normalize_for_compare(u["text"]) for u in units_old]
    texts_new = [normalize_for_compare(u["text"]) for u in units_new]

    # Edge case: no text
    if not texts_old and not texts_new:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False, indent=2)
        print("No text in both files.")
        return

    # Vectorize and similarity
    vectorizer = TfidfVectorizer(ngram_range=TFIDF_NGRAM).fit(texts_old + texts_new)
    tfidf_old = vectorizer.transform(texts_old) if texts_old else None
    tfidf_new = vectorizer.transform(texts_new) if texts_new else None
    sim_matrix = cosine_similarity(tfidf_old, tfidf_new) if (tfidf_old is not None and tfidf_new is not None) else np.zeros((len(texts_old), len(texts_new)))

    # Global matching (one-to-one candidate pairs)
    candidate_pairs = global_optimal_matching(sim_matrix, use_hungarian=use_hungarian)
    # sort desc by sim to accept best matches first
    candidate_pairs.sort(key=lambda x: x[2], reverse=True)

    # Keep only pairs >= match_threshold (ensuring one-to-one)
    matched_old = set()
    matched_new = set()
    matched_pairs = []
    for i, j, s in candidate_pairs:
        if i in matched_old or j in matched_new:
            continue
        if s >= match_threshold:
            matched_old.add(i); matched_new.add(j)
            matched_pairs.append((i, j, s))

    # detect split/merge
    splits, merges = detect_splits_and_merges(sim_matrix)

    mapping = []
    matched_old_idx = set([p[0] for p in matched_pairs])
    matched_new_idx = set([p[1] for p in matched_pairs])

    # 1) matched 1->1 => modified (skip unchanged)
    for i, j, s in matched_pairs:
        old_u = units_old[i]
        new_u = units_new[j]
        if normalize_for_compare(old_u["text"]) == normalize_for_compare(new_u["text"]):
            # unchanged -> skip (do not include in output)
            continue
        mapping.append({
            "unit_2013": {
                "article_id": old_u.get("article_id"),
                "article_number": old_u.get("article_number"),
                "clause_id": old_u.get("clause_id"),
                "point_id": old_u.get("point_id")
            },
            "unit_2024": {
                "article_id": new_u.get("article_id"),
                "article_number": new_u.get("article_number"),
                "clause_id": new_u.get("clause_id"),
                "point_id": new_u.get("point_id")
            },
            "similarity": round(float(s), 3),
            "change_type": "modified",
            "before_change": normalize_for_output(old_u.get("text", "")),
            "after_change": normalize_for_output(new_u.get("text", ""))
        })

    # 2) splits -> treat as DELETED (old unit)
    split_old_indices = set()
    for i, candidates in splits.items():
        # mark deleted for old unit i
        old_u = units_old[i]
        mapping.append({
            "unit_2013": {
                "article_id": old_u.get("article_id"),
                "article_number": old_u.get("article_number"),
                "clause_id": old_u.get("clause_id"),
                "point_id": old_u.get("point_id")
            },
            "unit_2024": None,
            "similarity": 0.0,
            "change_type": "deleted",
            "before_change": normalize_for_output(old_u.get("text", "")),
            "after_change": ""
        })
        split_old_indices.add(i)

    # 3) merges -> treat as ADDED (new unit)
    merge_new_indices = set()
    for j, candidates in merges.items():
        new_u = units_new[j]
        mapping.append({
            "unit_2013": None,
            "unit_2024": {
                "article_id": new_u.get("article_id"),
                "article_number": new_u.get("article_number"),
                "clause_id": new_u.get("clause_id"),
                "point_id": new_u.get("point_id")
            },
            "similarity": 0.0,
            "change_type": "added",
            "before_change": "",
            "after_change": normalize_for_output(new_u.get("text", ""))
        })
        merge_new_indices.add(j)

    # 4) deleted: old units not matched and not already handled by splits
    merge_old_indices = set(i for j,c in merges.items() for i,_ in c)
    for i, old_u in enumerate(units_old):
        if i in matched_old_idx:
            continue
        if i in split_old_indices:
            continue
        # if this old is part of a merge (i in merge_old_indices) -> we still mark deleted
        if i in merge_old_indices:
            mapping.append({
                "unit_2013": {
                    "article_id": old_u.get("article_id"),
                    "article_number": old_u.get("article_number"),
                    "clause_id": old_u.get("clause_id"),
                    "point_id": old_u.get("point_id")
                },
                "unit_2024": None,
                "similarity": 0.0,
                "change_type": "deleted",
                "before_change": normalize_for_output(old_u.get("text", "")),
                "after_change": ""
            })
            continue
        # otherwise standard deleted (no match)
        mapping.append({
            "unit_2013": {
                "article_id": old_u.get("article_id"),
                "article_number": old_u.get("article_number"),
                "clause_id": old_u.get("clause_id"),
                "point_id": old_u.get("point_id")
            },
            "unit_2024": None,
            "similarity": 0.0,
            "change_type": "deleted",
            "before_change": normalize_for_output(old_u.get("text", "")),
            "after_change": ""
        })

    # 5) added: new units not matched and not merge-handled (and skip new pieces of splits per your rule)
    split_new_indices = set(j for i,c in splits.items() for j,_ in c)
    for j, new_u in enumerate(units_new):
        if j in matched_new_idx:
            continue
        if j in merge_new_indices:
            continue
        if j in split_new_indices:
            # per user instruction: when split happens we mark old as deleted and skip listing split pieces as added
            continue
        mapping.append({
            "unit_2013": None,
            "unit_2024": {
                "article_id": new_u.get("article_id"),
                "article_number": new_u.get("article_number"),
                "clause_id": new_u.get("clause_id"),
                "point_id": new_u.get("point_id")
            },
            "similarity": 0.0,
            "change_type": "added",
            "before_change": "",
            "after_change": normalize_for_output(new_u.get("text", ""))
        })

    # Save only changed entries (unchanged were skipped earlier)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)

    # Summary print
    counts = {}
    for m in mapping:
        counts[m["change_type"]] = counts.get(m["change_type"], 0) + 1
    print(f"✅ Saved {len(mapping)} mappings to {output_file}")
    print("Summary:", counts)
    if _HUNGARIAN_AVAILABLE:
        print("Hungarian (linear_sum_assignment) used.")
    else:
        print("Hungarian not available; greedy matching used.")

# ----------------- Run example -----------------
if __name__ == "__main__":
    generate_mapping(
        "output/articles_2013.json",
        "output/articles_2024.json",
        "output/mapping.json",
        match_threshold=MATCH_THRESHOLD,
        use_hungarian=True
    )