#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""学术论文检索（零安装，仅用 Python 标准库）。

数据源：
  arxiv      arXiv 预印本（数学/科学计算最新工作）
  crossref   正式发表论文 + DOI（覆盖最全）
  openalex   综合检索，带引用数 / 开放获取 / 摘要（可替代 Semantic Scholar）

用法示例：
  python lit_search.py "numerical linear algebra" --source all --n 5
  python lit_search.py "Gauss quadrature error" --source arxiv
  python lit_search.py "finite element method" --source openalex --n 10

说明：
  - 依赖网络能访问 arxiv.org / api.crossref.org / api.openalex.org
  - Semantic Scholar 与 Google Scholar 在国内网络下被阻断，本脚本不含。
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

UA = {"User-Agent": "lit-search/1.0 (academic literature search; stdlib only)",
      "Accept": "*/*"}

# 强制 UTF-8 输出，避免中文在 Windows 终端乱码
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def http_get(url, retries=3):
    """带重试的 GET。arXiv 偶发 406/429/503，间隔重试即可恢复。"""
    last = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            last = e
            if e.code in (406, 429, 503):
                time.sleep(1 + i)
                continue
            raise
        except urllib.error.URLError as e:
            last = e
            time.sleep(1 + i)
    raise last


def _short(s, n=280):
    s = (s or "").replace("\n", " ").strip()
    return s if len(s) <= n else s[:n] + "…"


# ---------------------------------------------------------------- arXiv ----
def search_arxiv(query, n):
    q = urllib.parse.quote(query)
    url = (f"https://export.arxiv.org/api/query"
           f"?search_query=all:{q}&start=0&max_results={n}")
    ns = {"a": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(http_get(url))
    out = []
    for e in root.findall("a:entry", ns):
        pid = (e.findtext("a:id", "", ns) or "").strip()
        authors = [a.findtext("a:name", "", ns) for a in e.findall("a:author", ns)]
        out.append({
            "title": (e.findtext("a:title", "", ns) or "").strip().replace("\n", " "),
            "year": (e.findtext("a:published", "", ns) or "")[:4],
            "authors": authors,
            "id": pid.split("/abs/")[-1],
            "url": pid,
            "abstract": _short(e.findtext("a:summary", "", ns)),
        })
    return out


# ------------------------------------------------------------- Crossref ----
def search_crossref(query, n):
    q = urllib.parse.quote(query)
    url = (f"https://api.crossref.org/works?query.bibliographic={q}&rows={n}"
           f"&select=title,DOI,issued,author,container-title")
    data = json.loads(http_get(url))
    out = []
    for it in data["message"]["items"]:
        year = ""
        try:
            year = str(it["issued"]["date-parts"][0][0])
        except Exception:
            pass
        authors = [f"{a.get('given', '')} {a.get('family', '')}".strip()
                   for a in it.get("author", [])]
        doi = it.get("DOI", "")
        out.append({
            "title": (it.get("title") or [""])[0],
            "year": year,
            "authors": authors,
            "id": doi,
            "url": f"https://doi.org/{doi}" if doi else "",
            "venue": (it.get("container-title") or [""])[0],
        })
    return out


# -------------------------------------------------------------- OpenAlex ----
def _inverted_to_text(inv):
    if not inv:
        return ""
    pos = {}
    for word, idxs in inv.items():
        for i in idxs:
            pos[i] = word
    return " ".join(pos[i] for i in sorted(pos))


def search_openalex(query, n):
    q = urllib.parse.quote(query)
    sel = ("id,title,display_name,publication_year,doi,authorships,"
           "primary_location,cited_by_count,abstract_inverted_index,open_access")
    url = f"https://api.openalex.org/works?search={q}&per-page={n}&select={sel}"
    data = json.loads(http_get(url))
    out = []
    for w in data.get("results", []):
        authors = [a["author"]["display_name"] for a in w.get("authorships", [])]
        loc = w.get("primary_location") or {}
        pdf = loc.get("pdf_url") or loc.get("landing_page_url") or ""
        doi = (w.get("doi") or "").replace("https://doi.org/", "")
        out.append({
            "title": w.get("title") or w.get("display_name", ""),
            "year": str(w.get("publication_year") or ""),
            "authors": authors,
            "id": doi,
            "url": w.get("doi") or w.get("id", ""),
            "citations": w.get("cited_by_count", 0),
            "open_access": (w.get("open_access") or {}).get("oa_status", ""),
            "pdf": pdf,
            "abstract": _short(_inverted_to_text(w.get("abstract_inverted_index"))),
        })
    return out


# ----------------------------------------------------------------- print ----
def render(papers):
    for i, p in enumerate(papers, 1):
        extra = []
        if p.get("venue"):
            extra.append(p["venue"])
        if p.get("citations") is not None:
            extra.append(f"被引 {p['citations']}")
        if p.get("open_access"):
            extra.append(f"OA:{p['open_access']}")
        line = f"[{i}] {p['title']} ({p.get('year') or '?'})"
        if p.get("authors"):
            line += f"\n    作者: {', '.join(p['authors'][:6])}"
        if extra:
            line += f"\n    {' | '.join(extra)}"
        if p.get("id"):
            line += f"\n    id: {p['id']}"
        if p.get("url"):
            line += f"\n    url: {p['url']}"
        if p.get("pdf"):
            line += f"\n    pdf: {p['pdf']}"
        if p.get("abstract"):
            line += f"\n    摘要: {p['abstract']}"
        print(line)
        print()


SOURCES = {
    "arxiv": search_arxiv,
    "crossref": search_crossref,
    "openalex": search_openalex,
}


def main():
    ap = argparse.ArgumentParser(description="学术论文检索（arXiv/Crossref/OpenAlex）")
    ap.add_argument("query", help="检索词，如 \"numerical linear algebra\"")
    ap.add_argument("--source", choices=list(SOURCES) + ["all"],
                    default="all", help="数据源，默认 all")
    ap.add_argument("--n", type=int, default=5, help="每个源返回条数，默认 5")
    args = ap.parse_args()

    sources = list(SOURCES) if args.source == "all" else [args.source]
    for name in sources:
        print(f"========== {name.upper()} ==========")
        try:
            papers = SOURCES[name](args.query, args.n)
            if papers:
                render(papers)
            else:
                print("（无结果）")
        except Exception as e:
            print(f"（{name} 检索失败: {e}）")
        print()


if __name__ == "__main__":
    main()
