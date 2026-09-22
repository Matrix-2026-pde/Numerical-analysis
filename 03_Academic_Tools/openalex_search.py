"""
OpenAlex API wrapper for the ``works`` endpoint.

Two search modes are supported:
  * ``search``        -- full-text relevance search (OpenAlex ``search=`` param,
                         translated to ``fulltext.search`` on the server).
  * ``title_filter``  -- search *only* within the title (``filter=title.search:``).

A real ``mailto`` is always sent so that requests enter OpenAlex's "polite pool"
(faster and higher rate limits).

Example
-------
    results, meta = search_works(title_filter="Jacobi iteration",
                                 sort="cited_by_count:desc", per_page=5)
"""

import requests

BASE_URL = "https://api.openalex.org/works"
MAILTO = "owen-0706@outlook.com"          # real email -> polite pool


def search_works(search=None, title_filter=None, sort=None, per_page=25,
                 mailto=MAILTO, timeout=30):
    """Query the OpenAlex ``works`` endpoint.

    Parameters
    ----------
    search : str, optional
        Full-text search phrase (relevance ranked by the server).
    title_filter : str, optional
        Phrase to search for in the title only.
    sort : str, optional
        Sort specifier, e.g. ``"cited_by_count:desc"`` or ``"publication_date:desc"``.
    per_page : int
        Number of results per page (max 200).
    mailto : str
        Contact email for the polite pool.
    timeout : float
        Request timeout in seconds.

    Returns
    -------
    (results, meta) : (list, dict)
        ``results`` is the list of work objects; ``meta`` contains the query
        metadata (total count, page, per-page, etc.).
    """
    params = {
        "mailto": mailto,
        "per-page": per_page,
    }
    if search:
        params["search"] = search
    if title_filter:
        params["filter"] = f"title.search:{title_filter}"
    if sort:
        params["sort"] = sort

    resp = requests.get(BASE_URL, params=params, timeout=timeout)
    resp.raise_for_status()                 # raise on HTTP errors (4xx/5xx)
    data = resp.json()
    return data.get("results", []), data.get("meta", {})


def _fmt(value, width=16):
    """Format a possibly-None value for the summary table."""
    if value is None:
        value = "—"
    s = str(value)
    return s if len(s) <= width else s[: width - 1] + "…"


def main():
    # Test: top-5 works whose *title* mentions "Jacobi iteration",
    # ranked by citation count.
    results, meta = search_works(
        title_filter="Jacobi iteration",
        sort="cited_by_count:desc",
        per_page=5,
    )

    print("=" * 78)
    print("OpenAlex works search (title filter)")
    print("=" * 78)
    print(f"query        : title.search:Jacobi iteration")
    print(f"sort         : cited_by_count:desc")
    print(f"total matches: {meta.get('count')}")
    print(f"shown        : {len(results)}")
    print("-" * 78)

    for i, w in enumerate(results, 1):
        title = w.get("title") or w.get("display_name") or "(no title)"
        year = w.get("publication_year")
        cited = w.get("cited_by_count")
        doi = (w.get("doi") or "n/a").replace("https://doi.org/", "")
        print(f"[{i}] {title}")
        print(f"    year={year}  cited_by={cited}  doi={doi}")

    print("-" * 78)
    print("Done.")


if __name__ == "__main__":
    main()
