"""Stage 1: Download HTML for papers that have reviewed equations."""


import concurrent.futures
import hashlib
import json
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

from config import (
    ANNOTATIONS_FILE,
    ARXIV_HTML_URL,
    DOWNLOAD_BACKOFF_BASE,
    DOWNLOAD_MAX_RETRIES,
    DOWNLOAD_MAX_WORKERS,
    DOWNLOAD_REPORT,
    DOWNLOAD_TIMEOUT,
    EQUATIONS_FILE,
    HTML_DIR,
    OUTPUT_DIR,
)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}


def _paper_ids_with_equations() -> list[str]:
    equations = json.loads(EQUATIONS_FILE.read_text(encoding="utf-8"))
    return [pid for pid, entries in equations.items() if entries]


def _is_valid_html(path: Path) -> bool:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
        if len(text) < 500:
            return False
        soup = BeautifulSoup(text, "html.parser")
        return bool(soup.find("body"))
    except Exception:
        return False


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _download_one(paper_id: str) -> dict:
    url = ARXIV_HTML_URL.format(paper_id=paper_id)
    dest = HTML_DIR / f"{paper_id}.html"

    if dest.exists() and _is_valid_html(dest):
        return {
            "paper_id": paper_id,
            "status": "cached",
            "url": url,
            "attempts": 0,
            "size": dest.stat().st_size,
            "sha256": _sha256(dest),
            "failure_reason": None,
        }

    attempt = 0
    failure_reason = None
    for attempt in range(1, DOWNLOAD_MAX_RETRIES + 1):
        try:
            resp = requests.get(url, timeout=DOWNLOAD_TIMEOUT, headers=_HEADERS)
            if resp.status_code == 200:
                tmp = dest.with_suffix(".html.tmp")
                tmp.write_text(resp.text, encoding="utf-8")
                if not _is_valid_html(tmp):
                    tmp.unlink(missing_ok=True)
                    failure_reason = "invalid_html"
                    break
                tmp.replace(dest)
                return {
                    "paper_id": paper_id,
                    "status": "downloaded",
                    "url": url,
                    "attempts": attempt,
                    "size": dest.stat().st_size,
                    "sha256": _sha256(dest),
                    "failure_reason": None,
                }
            if resp.status_code == 429:
                retry_after = float(resp.headers.get("Retry-After", DOWNLOAD_BACKOFF_BASE ** attempt))
                time.sleep(min(retry_after, 60.0))
                continue
            if resp.status_code >= 500:
                time.sleep(DOWNLOAD_BACKOFF_BASE ** attempt)
                continue
            failure_reason = f"http_{resp.status_code}"
            break
        except requests.Timeout:
            failure_reason = "timeout"
            time.sleep(DOWNLOAD_BACKOFF_BASE ** attempt)
        except Exception as exc:
            failure_reason = f"{type(exc).__name__}: {exc}"
            break

    return {
        "paper_id": paper_id,
        "status": "failed",
        "url": url,
        "attempts": attempt,
        "size": None,
        "sha256": None,
        "failure_reason": failure_reason,
    }


def run() -> dict:
    HTML_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    paper_ids = _paper_ids_with_equations()
    results: list[dict] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=DOWNLOAD_MAX_WORKERS) as pool:
        futures = {pool.submit(_download_one, pid): pid for pid in paper_ids}
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())

    results.sort(key=lambda r: r["paper_id"])
    by_status: dict[str, list[str]] = {}
    for r in results:
        by_status.setdefault(r["status"], []).append(r["paper_id"])

    report = {
        "total": len(paper_ids),
        "cached": len(by_status.get("cached", [])),
        "downloaded": len(by_status.get("downloaded", [])),
        "failed": len(by_status.get("failed", [])),
        "results": results,
    }

    tmp = DOWNLOAD_REPORT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(DOWNLOAD_REPORT)

    return report
