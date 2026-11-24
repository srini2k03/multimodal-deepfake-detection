"""Check the licence and access conditions of candidate datasets and record the evidence.

Runs in GitHub Actions (the dev sandbox cannot reach these hosts). Standard library only.
For every candidate it records what the host reports right now: HTTP status of the
landing page / API, gating flags, the stated licence, file sizes and whether anonymous
range requests work. The JSON output is committed by ``.github/workflows/dataset-access.yml``
and summarised in ``docs/DATASETS.md``.

    python scripts/check_dataset_access.py --out ../results/datasets/access_check.json
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

UA = "dataset-access-check/1.0 (+https://github.com/srini2k03/multimodal-deepfake-detection)"
TIMEOUT = 60

# Hugging Face dataset repos that have been proposed for face-forgery / AV-deepfake work.
HF_CANDIDATES = [
    "OpenRL/DeepFakeFace",
    "xingjunm/WildDeepfake",
    "ControlNet/LAV-DF",
    "ComplexDataLab/OpenFake",
    "mueller91/MLAAD",
    "elsaEU/ELSA_D3",
    "Hemg/deepfake-and-real-images",
]
HF_TREES = ["xingjunm/WildDeepfake", "rshaojimmy/Seq-DeepFake"]
HF_SEARCH_TERMS = ["deepfake", "face forgery", "fake face", "face swap", "synthetic faces", "audio deepfake"]

# Zenodo records: OpenForensics, WaveFake, ASVspoof 2021 LA / DF evaluation parts.
ZENODO_RECORDS = {
    "OpenForensics": "5528418",
    "WaveFake": "5642694",
    "ASVspoof2021_LA_eval": "4837263",
    "ASVspoof2021_DF_eval_part00": "4835108",
}
ZENODO_SEARCH = ["deepfake faces dataset", "face forgery dataset", "in-the-wild audio deepfake"]

DATASHARE_HANDLE = "https://datashare.ed.ac.uk/handle/10283/3336"  # ASVspoof 2019 (LA + PA)
DATASHARE_API = "https://datashare.ed.ac.uk/server/api"
EXTRA_PAGES = {
    "in_the_wild_audio": "https://deepfake-total.com/in_the_wild",
    "asvspoof2021_keys": "https://www.asvspoof.org/asvspoof2021/LA-keys-full.tar.gz",
}


def _request(url: str, method: str = "GET", headers: dict[str, str] | None = None, read: int | None = 0):
    req = urllib.request.Request(url, method=method, headers={"User-Agent": UA, **(headers or {})})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            body = b""
            if read is None:
                body = resp.read()
            elif read > 0:
                body = resp.read(read)
            return {
                "status": resp.status,
                "url": resp.geturl(),
                "headers": {k.lower(): v for k, v in resp.headers.items()},
                "body": body,
                "seconds": round(time.perf_counter() - t0, 3),
            }
    except urllib.error.HTTPError as exc:
        return {"status": exc.code, "url": url, "headers": dict(exc.headers or {}), "body": b"", "error": str(exc)}
    except (OSError, ValueError) as exc:  # network errors are data here, not failures
        return {"status": None, "url": url, "headers": {}, "body": b"", "error": f"{type(exc).__name__}: {exc}"}


def _json(url: str) -> tuple[int | None, Any]:
    r = _request(url, read=None)
    if r["status"] != 200:
        return r["status"], {"error": r.get("error")}
    try:
        return 200, json.loads(r["body"])
    except json.JSONDecodeError:
        return 200, {"error": "invalid json"}


def _range_probe(url: str, nbytes: int = 32 * 1024 * 1024) -> dict[str, Any]:
    """Download the first ``nbytes`` with a Range header; report status, size and throughput."""
    r = _request(url, headers={"Range": f"bytes=0-{nbytes - 1}"}, read=None)
    got = len(r["body"])
    out = {
        "status": r["status"],
        "final_url": r["url"],
        "content_range": r["headers"].get("content-range"),
        "content_type": r["headers"].get("content-type"),
        "bytes": got,
        "seconds": r.get("seconds"),
        "error": r.get("error"),
    }
    if got and r.get("seconds"):
        out["mb_per_s"] = round(got / 1e6 / r["seconds"], 2)
    if got >= 4:
        out["magic"] = r["body"][:4].hex()
    return out


# --------------------------------------------------------------------------- Hugging Face
def hf_dataset(repo: str) -> dict[str, Any]:
    status, info = _json(f"https://huggingface.co/api/datasets/{repo}")
    out: dict[str, Any] = {"repo": repo, "api_status": status}
    if status != 200:
        out["error"] = info.get("error")
        return out
    card = info.get("cardData") or {}
    out.update(
        gated=info.get("gated"),
        private=info.get("private"),
        disabled=info.get("disabled"),
        license=card.get("license"),
        license_tags=[t for t in info.get("tags", []) if t.startswith("license:")],
        size_tags=[t for t in info.get("tags", []) if t.startswith("size_categories:")],
        used_storage_bytes=info.get("usedStorage"),
        n_files=len(info.get("siblings") or []),
        files_sample=[s.get("rfilename") for s in (info.get("siblings") or [])[:25]],
        last_modified=info.get("lastModified"),
        downloads=info.get("downloads"),
    )
    status, tree = _json(f"https://huggingface.co/api/datasets/{repo}/tree/main")
    if status == 200 and isinstance(tree, list):
        out["top_level"] = [
            {"path": e.get("path"), "type": e.get("type"), "size": (e.get("lfs") or {}).get("size", e.get("size"))}
            for e in tree[:40]
        ]
    readme = _request(f"https://huggingface.co/datasets/{repo}/raw/main/README.md", read=6000)
    out["readme_status"] = readme["status"]
    if readme["status"] == 200:
        text = readme["body"].decode("utf-8", "replace")
        out["readme_excerpt"] = text[:2500]
        out["readme_licence_lines"] = [
            ln.strip()[:300] for ln in text.splitlines() if re.search(r"licen[cs]e|terms|non-commercial", ln, re.I)
        ][:15]
    return out


def hf_tree(repo: str, path: str = "", depth: int = 2) -> list[dict[str, Any]]:
    """List a dataset repo directory (and sub-directories up to ``depth``) with file sizes."""
    status, tree = _json(f"https://huggingface.co/api/datasets/{repo}/tree/main/{urllib.parse.quote(path)}")
    if status != 200 or not isinstance(tree, list):
        return [{"path": path, "error": f"status {status}"}]
    out: list[dict[str, Any]] = []
    for e in tree:
        size = (e.get("lfs") or {}).get("size", e.get("size"))
        if e.get("type") == "directory" and depth > 0:
            children = hf_tree(repo, e["path"], depth - 1)
            files = [c for c in children if c.get("type") == "file"]
            out.append(
                {
                    "path": e["path"],
                    "type": "directory",
                    "n_files": len(files),
                    "bytes": sum(int(c.get("size") or 0) for c in files),
                    "examples": [c["path"] for c in files[:5]],
                    "subdirs": [c for c in children if c.get("type") == "directory"],
                }
            )
        else:
            out.append({"path": e.get("path"), "type": e.get("type"), "size": size})
    return out


def hf_search(term: str) -> list[dict[str, Any]]:
    q = urllib.parse.urlencode({"search": term, "limit": 60, "full": "true"})
    status, items = _json(f"https://huggingface.co/api/datasets?{q}")
    if status != 200 or not isinstance(items, list):
        return [{"error": f"status {status}"}]
    return [
        {
            "id": d.get("id"),
            "gated": d.get("gated"),
            "license": [t for t in d.get("tags", []) if t.startswith("license:")],
            "size": [t for t in d.get("tags", []) if t.startswith("size_categories:")],
            "downloads": d.get("downloads"),
        }
        for d in items
    ]


# --------------------------------------------------------------------------- Zenodo
def zenodo_record(record_id: str) -> dict[str, Any]:
    status, rec = _json(f"https://zenodo.org/api/records/{record_id}")
    out: dict[str, Any] = {"record": record_id, "api_status": status}
    if status != 200:
        out["error"] = rec.get("error")
        return out
    meta = rec.get("metadata", {})
    files = rec.get("files") or []
    out.update(
        title=meta.get("title"),
        access_right=meta.get("access_right"),
        license=(meta.get("license") or {}).get("id") if isinstance(meta.get("license"), dict) else meta.get("license"),
        publication_date=meta.get("publication_date"),
        doi=rec.get("doi"),
        files=[{"key": f.get("key"), "size": f.get("size")} for f in files[:40]],
        total_bytes=sum(int(f.get("size") or 0) for f in files),
    )
    if files:
        link = (files[0].get("links") or {}).get("self")
        if link:
            out["range_probe_first_file"] = _range_probe(link, 4 * 1024 * 1024)
    return out


def zenodo_search(term: str) -> list[dict[str, Any]]:
    q = urllib.parse.urlencode({"q": term, "size": 15, "access_right": "open", "type": "dataset"})
    status, res = _json(f"https://zenodo.org/api/records?{q}")
    if status != 200:
        return [{"error": f"status {status}"}]
    hits = (res.get("hits") or {}).get("hits") or []
    return [
        {
            "id": h.get("id"),
            "title": (h.get("metadata") or {}).get("title"),
            "license": ((h.get("metadata") or {}).get("license") or {}).get("id"),
            "total_bytes": sum(int(f.get("size") or 0) for f in h.get("files") or []),
        }
        for h in hits
    ]


# --------------------------------------------------------------------------- DataShare (ASVspoof 2019)
def datashare() -> dict[str, Any]:
    out: dict[str, Any] = {"handle": DATASHARE_HANDLE}
    page = _request(DATASHARE_HANDLE, read=None)
    out["page_status"] = page["status"]
    html = page["body"].decode("utf-8", "replace")
    out["page_bytes"] = len(html)
    links = sorted(set(re.findall(r'href="([^"]*bitstream[^"]*)"', html)))
    out["bitstream_links"] = links[:30]
    licence_re = r"[^<>]{0,160}(?:licen[cs]e|ODC|Open Data Commons)[^<>]{0,160}"
    out["licence_lines"] = sorted({m.strip()[:300] for m in re.findall(licence_re, html, re.I)})[:20]
    # DSpace 7 REST: resolve the handle, then list bitstreams with their sizes.
    status, item = _json(f"{DATASHARE_API}/pid/find?id=hdl:10283/3336")
    out["rest_status"] = status
    if status == 200 and isinstance(item, dict) and "uuid" in item:
        md = item.get("metadata") or {}
        out["rest_rights"] = {k: [v.get("value") for v in vals] for k, vals in md.items() if k.startswith("dc.rights")}
        out["rest_title"] = [v.get("value") for v in md.get("dc.title", [])]
        bundles_status, bundles = _json(f"{DATASHARE_API}/core/items/{item['uuid']}/bundles?size=20")
        out["bundles_status"] = bundles_status
        files = []
        for b in ((bundles or {}).get("_embedded") or {}).get("bundles", []):
            href = ((b.get("_links") or {}).get("bitstreams") or {}).get("href")
            if not href:
                continue
            _, bs = _json(href + "?size=50")
            for f in ((bs or {}).get("_embedded") or {}).get("bitstreams", []):
                files.append(
                    {
                        "bundle": b.get("name"),
                        "name": f.get("name"),
                        "size": f.get("sizeBytes"),
                        "content": ((f.get("_links") or {}).get("content") or {}).get("href"),
                    }
                )
        out["rest_files"] = files
    # Candidate direct URLs for LA.zip (DSpace 6 and 7 style); probe range access + throughput.
    candidates = [urllib.parse.urljoin(DATASHARE_HANDLE, link) for link in links if "LA.zip" in link]
    candidates += [f["content"] for f in out.get("rest_files", []) if f.get("name") == "LA.zip" and f.get("content")]
    candidates.append("https://datashare.ed.ac.uk/bitstream/handle/10283/3336/LA.zip")
    probes = []
    for url in dict.fromkeys(candidates):
        head = _request(url, method="HEAD")
        probe = {"url": url, "head_status": head["status"], "content_length": head["headers"].get("content-length")}
        probe["head_final_url"] = head["url"]
        probe["range"] = _range_probe(url)
        probes.append(probe)
        if probe["range"].get("status") in (200, 206) and probe["range"].get("bytes"):
            break
    out["la_zip_probes"] = probes
    return out


def extra_pages() -> dict[str, Any]:
    out = {}
    for name, url in EXTRA_PAGES.items():
        head = _request(url, method="HEAD")
        page = _request(url, read=20000) if head["status"] == 200 else head
        text = page["body"].decode("utf-8", "replace") if page.get("body") else ""
        out[name] = {
            "url": url,
            "status": head["status"],
            "final_url": head["url"],
            "content_length": head["headers"].get("content-length"),
            "content_type": head["headers"].get("content-type"),
            "licence_lines": sorted(
                {m.strip()[:300] for m in re.findall(r"[^<>]{0,160}licen[cs]e[^<>]{0,160}", text, re.I)}
            )[:10],
            "download_links": sorted(set(re.findall(r'href="([^"]+\.(?:zip|tar\.gz|tgz))"', text)))[:10],
            "all_links": sorted(set(re.findall(r'href="(https?://[^"]+)"', text)))[:40],
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    report: dict[str, Any] = {"checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    report["asvspoof2019_datashare"] = datashare()
    report["huggingface"] = [hf_dataset(r) for r in HF_CANDIDATES]
    report["huggingface_search"] = {t: hf_search(t) for t in HF_SEARCH_TERMS}
    report["huggingface_trees"] = {r: hf_tree(r) for r in HF_TREES}
    report["zenodo"] = {name: zenodo_record(rid) for name, rid in ZENODO_RECORDS.items()}
    report["zenodo_search"] = {t: zenodo_search(t) for t in ZENODO_SEARCH}
    report["other"] = extra_pages()
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=1, default=str) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=1, default=str)[:60000])


if __name__ == "__main__":
    main()
