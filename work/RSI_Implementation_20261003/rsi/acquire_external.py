"""Acquire/audit official ISIC Task1 and PH2 files; never score a model.

Downloads preserve resumable .part files, official HTTP metadata, and local
SHA256. S3 ETags are recorded as opaque metadata, not claimed source checksums.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import re
import shutil
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlparse

import requests
from PIL import Image


PROJECT = Path(__file__).resolve().parents[1]
OFFICIAL_ISIC = "https://challenge.isic-archive.com/data/"
OFFICIAL_PH2 = "https://www.fc.up.pt/addi/ph2%20database.html"
PH2_URLS = ["https://www.dropbox.com/s/k88qukc20ljnbuo/PH2Dataset.rar?dl=1",
            "https://dl.dropboxusercontent.com/s/k88qukc20ljnbuo/PH2Dataset.rar"]
EXPECTED = {"ISIC2017": {"train": 2000, "val": 150, "test": 600},
            "ISIC2018": {"train": 2594, "val": 100, "test": 1000}, "PH2": {"external": 200}}
LOCK = threading.RLock()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def save_status(status: dict) -> None:
    with LOCK:
        status["updated_utc"] = utc()
        write_json(PROJECT / "outputs/external_data_status.json", status)


def discover_sources(status: dict) -> list[dict]:
    response = requests.get(OFFICIAL_ISIC, timeout=(30, 90))
    response.raise_for_status()
    (PROJECT / "data_manifests/external_official_isic_page.html").write_text(response.text)
    urls = list(dict.fromkeys(re.findall(r'href=["\']([^"\']+)["\']', response.text)))
    jobs = []
    for url in urls:
        name = Path(urlparse(url).path).name
        is2017 = "ISIC-2017_" in name and ("Part1_GroundTruth.zip" in name or name.endswith("_Data.zip"))
        is2018 = ("ISIC2018_Task1-2_" in name and name.endswith("_Input.zip")) or ("ISIC2018_Task1_" in name and name.endswith("_GroundTruth.zip"))
        if not (is2017 or is2018):
            continue
        split = "train" if "Training" in name else "val" if "Validation" in name else "test"
        jobs.append({"dataset": "ISIC2017" if is2017 else "ISIC2018", "split": split,
                     "kind": "masks" if "GroundTruth" in name else "images", "filename": name,
                     "source_url": url, "official_page": OFFICIAL_ISIC, "state": "pending"})
    if len(jobs) != 12:
        raise ValueError(f"Expected12 official Task1 bundles, found{len(jobs)}")
    jobs.append({"dataset": "PH2", "split": "external", "kind": "archive", "filename": "PH2Dataset.rar",
                 "source_url": PH2_URLS[0], "alternative_official_url": PH2_URLS[1],
                 "official_page": OFFICIAL_PH2, "state": "pending"})
    status["sources"] = jobs
    status["official_page_snapshot_sha256"] = hashlib.sha256(response.content).hexdigest()
    write_json(PROJECT / "data_manifests/external_official_sources.json", {"retrieved_utc": utc(), "sources": jobs})
    return jobs


def download(job: dict, root: Path, status: dict, attempts: int = 5) -> Path:
    directory = root / job["dataset"].lower() / "downloads"
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / job["filename"]
    partial = target.with_suffix(target.suffix + ".part")
    job["local_archive"] = str(target)
    urls = [job["source_url"]] + ([job["alternative_official_url"]] if "alternative_official_url" in job else [])
    job["failures"] = []
    for url in urls:
        expected = None
        head = None
        try:
            head = requests.head(url, allow_redirects=True, timeout=(20, 40))
            head.raise_for_status()
            expected = int(head.headers["Content-Length"]) if head.headers.get("Content-Length", "").isdigit() else None
            job.update(resolved_url=head.url, head_http_status=head.status_code, http_etag=head.headers.get("ETag"),
                       http_content_length=expected, http_last_modified=head.headers.get("Last-Modified"),
                       official_checksum=None, checksum_note="No independent published MD5/SHA checksum supplied; ETag is opaque HTTP metadata.")
        except Exception as exc:
            job["failures"].append({"url": url, "method": "HEAD", "error": repr(exc), "utc": utc()})
        if target.exists() and (expected is None or target.stat().st_size == expected):
            job.update(state="downloaded", bytes=target.stat().st_size, local_sha256=sha256(target), cached=True)
            save_status(status)
            return target
        for attempt in range(attempts if job["dataset"] != "PH2" else 2):
            try:
                offset = partial.stat().st_size if partial.exists() else 0
                headers = {"Range": f"bytes={offset}-", "Accept-Encoding": "identity"} if offset else {"Accept-Encoding": "identity"}
                if offset and job.get("http_etag"):
                    headers["If-Range"] = job["http_etag"]
                job.update(state="downloading", current_url=url, downloaded_bytes=offset, attempt=attempt + 1)
                save_status(status)
                started = time.monotonic()
                last_report = started
                with requests.get(url, headers=headers, stream=True, timeout=(20, 90)) as response:
                    response.raise_for_status()
                    if "text/html" in response.headers.get("Content-Type", ""):
                        raise ValueError("Download returned HTML rather than the official archive")
                    resume = offset > 0 and response.status_code == 206
                    if resume and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                        raise ValueError("Resume Content-Range does not match the retained partial bytes")
                    if not resume:
                        offset = 0
                    total = response.headers.get("Content-Range", "").split("/")[-1]
                    content_length = response.headers.get("Content-Length")
                    expected = int(total) if total.isdigit() else offset + int(content_length) if content_length and content_length.isdigit() else expected
                    job.update(resolved_url=response.url, get_http_status=response.status_code,
                               http_etag=response.headers.get("ETag", job.get("http_etag")), http_content_length=expected)
                    with partial.open("ab" if resume else "wb") as stream:
                        for chunk in response.iter_content(4 * 1024 * 1024):
                            if chunk:
                                stream.write(chunk)
                            now = time.monotonic()
                            if now - last_report >= 20:
                                stream.flush()
                                count = partial.stat().st_size
                                job["downloaded_bytes"] = count
                                save_status(status)
                                print("download", job["filename"], count, "/", expected,
                                      "MiB/s", round((count-offset)/(now-started)/1048576, 2), flush=True)
                                last_report = now
                if expected is not None and partial.stat().st_size != expected:
                    raise ValueError(f"HTTP size mismatch: {partial.stat().st_size} != {expected}")
                partial.replace(target)
                job.update(state="downloaded", bytes=target.stat().st_size, local_sha256=sha256(target),
                           finished_utc=utc(), official_checksum=None)
                save_status(status)
                print("downloaded", target.name, target.stat().st_size, flush=True)
                return target
            except Exception as exc:
                failure = {"url": url, "method": "GET", "attempt": attempt + 1, "error": repr(exc), "utc": utc()}
                job["failures"].append(failure)
                save_status(status)
                print("download failure", target.name, repr(exc), flush=True)
                time.sleep(min(2 ** attempt, 10))
    job["state"] = "failed"
    save_status(status)
    raise RuntimeError(f"All official download attempts failed: {job['filename']}")


def extract_zip(job: dict, archive: Path, root: Path, status: dict) -> None:
    destination = root / job["dataset"].lower() / job["split"] / job["kind"]
    destination.mkdir(parents=True, exist_ok=True)
    selected = {}
    ignored = 0
    with zipfile.ZipFile(archive) as bundle:
        for item in bundle.infolist():
            name = Path(item.filename).name
            image = job["kind"] == "images" and re.fullmatch(r"ISIC_\d+\.(jpg|jpeg)", name, flags=re.I)
            mask = job["kind"] == "masks" and re.fullmatch(r"ISIC_\d+_segmentation\.png", name, flags=re.I)
            if not (image or mask) or "__MACOSX" in Path(item.filename).parts:
                ignored += int(not item.is_dir())
                continue
            if name in selected:
                raise ValueError(f"Duplicate selected ZIP filename: {name}")
            selected[name] = {"member": item.filename, "uncompressed_bytes": item.file_size, "zip_crc32": f"{item.CRC:08x}"}
            path = destination / name
            if not path.exists() or path.stat().st_size != item.file_size:
                temporary = path.with_suffix(path.suffix + ".part")
                with bundle.open(item) as source, temporary.open("wb") as out:
                    shutil.copyfileobj(source, out, length=4 * 1024 * 1024)
                temporary.replace(path)
    job.update(state="extracted", extracted_directory=str(destination), selected_files=len(selected), ignored_files=ignored,
               extraction="Task1 JPEG RGB or segmentation PNG only; ZIP CRC checked on each extracted member")
    write_json(PROJECT / f"data_manifests/external_{job['dataset'].lower()}_{job['split']}_{job['kind']}_members.json", selected)
    save_status(status)
    print("extracted", job["filename"], len(selected), flush=True)


def extract_ph2(archive: Path, root: Path, job: dict, status: dict) -> None:
    import rarfile
    destination = root / "ph2/external"
    selected = []
    with rarfile.RarFile(archive) as bundle:
        for item in bundle.infolist():
            member = item.filename.replace("\\", "/")
            name = Path(member).name
            if not name.lower().endswith(".bmp"):
                continue
            if "Dermoscopic_Image" in member:
                kind = "images"
            elif "lesion" in member.lower() and "roi" not in member.lower():
                kind = "masks"
            else:
                continue
            path = destination / kind / name
            path.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(item) as source, path.open("wb") as out:
                shutil.copyfileobj(source, out)
            selected.append(member)
    job.update(state="extracted", selected_files=len(selected), extraction="Original dermoscopic BMP + lesion BMP only")
    write_json(PROJECT / "data_manifests/external_ph2_members.json", selected)
    save_status(status)


def sample_id(path: Path) -> str:
    return re.sub(r"(_segmentation|_lesion|_mask)$", "", path.stem, flags=re.I)


def inspect_pair(dataset: str, split: str, image: Path, mask: Path) -> dict:
    with Image.open(image) as source:
        source.load()
        rgb = source.convert("RGB")
        dimensions = rgb.size
        rgb_pixel_hash = hashlib.sha256(rgb.tobytes()).hexdigest()
    with Image.open(mask) as source:
        source.load()
        mask_dimensions = source.size
        mode = source.mode
        decoded = source.convert("L")
        values = [i for i, count in enumerate(decoded.histogram()) if count]
        binary = set(values).issubset({0, 255}) or set(values).issubset({0, 1})
        mask_pixels_hash = hashlib.sha256(decoded.tobytes()).hexdigest()
        empty = not any(value > 0 for value in values)
    if dimensions != mask_dimensions:
        raise ValueError(f"image/mask size mismatch {dimensions} vs {mask_dimensions}")
    if not binary:
        raise ValueError(f"mask is not binary: {values}")
    return {"dataset": dataset, "split": split, "image_id": sample_id(image), "image_path": str(image),
            "mask_path": str(mask), "width": dimensions[0], "height": dimensions[1],
            "rgb_sha256": sha256(image), "rgb_pixel_sha256": rgb_pixel_hash, "mask_sha256": sha256(mask),
            "mask_pixel_sha256": mask_pixels_hash, "mask_mode": mode, "mask_values": json.dumps(values),
            "mask_empty": empty, "decode_passed": True, "dimensions_match": True, "mask_binary": binary}


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def audit_dataset(dataset: str, root: Path, status: dict, workers: int) -> list[dict]:
    all_rows, errors, split_counts = [], [], {}
    for split, expected in EXPECTED[dataset].items():
        directory = root / dataset.lower() / split
        images = {sample_id(p): p for p in (directory / "images").glob("*") if p.is_file() and not p.name.endswith(".part")}
        masks = {sample_id(p): p for p in (directory / "masks").glob("*") if p.is_file() and not p.name.endswith(".part")}
        unmatched_images, unmatched_masks = sorted(set(images) - set(masks)), sorted(set(masks) - set(images))
        count = {"expected": expected, "images": len(images), "masks": len(masks),
                 "unmatched_images": unmatched_images, "unmatched_masks": unmatched_masks, "decoded_pairs": 0}
        split_counts[split] = count
        pairs = [(dataset, split, images[key], masks[key]) for key in sorted(images.keys() & masks.keys())]
        with ThreadPoolExecutor(workers) as pool:
            futures = {pool.submit(inspect_pair, *pair): pair for pair in pairs}
            for index, future in enumerate(as_completed(futures), 1):
                try:
                    all_rows.append(future.result())
                    count["decoded_pairs"] += 1
                except Exception as exc:
                    errors.append({"dataset": dataset, "split": split, "image_id": sample_id(futures[future][2]), "error": repr(exc)})
                if index % 100 == 0:
                    print("audited", dataset, split, index, "/", len(pairs), flush=True)
        count["complete"] = len(images) == len(masks) == count["decoded_pairs"] == expected and not unmatched_images and not unmatched_masks
    all_rows.sort(key=lambda row: (row["split"], row["image_id"]))
    fields = ["dataset", "split", "image_id", "image_path", "mask_path", "width", "height", "rgb_sha256",
              "rgb_pixel_sha256", "mask_sha256", "mask_pixel_sha256", "mask_mode", "mask_values", "mask_empty",
              "decode_passed", "dimensions_match", "mask_binary"]
    write_csv(PROJECT / f"data_manifests/external_{dataset.lower()}_manifest.csv", all_rows, fields)
    write_json(PROJECT / f"data_manifests/external_{dataset.lower()}_audit_errors.json", errors)
    status["datasets"][dataset] = {"split_counts": split_counts, "all_files_passed": all(item["complete"] for item in split_counts.values()),
                                     "audit_errors": errors, "protocol": "Single-reference official Task1; independent initialization/training; no model scoring performed",
                                     "manifest": str(PROJECT / f"data_manifests/external_{dataset.lower()}_manifest.csv")}
    save_status(status)
    return all_rows


def overlap_audit(rows: list[dict], status: dict) -> None:
    ima_path = PROJECT / "data_manifests/final_references.csv"
    ima_info = {"state": "pending", "manifest": str(ima_path)}
    if ima_path.exists():
        ima_rows = {}
        for record in csv.DictReader(ima_path.open()):
            if record.get("audit_complete", "").lower() != "true":
                continue
            ima_rows.setdefault(record["image_id"], {"dataset": "IMA", "split": record["split"],
                                                     "image_id": record["image_id"], "rgb_sha256": record["image_sha256"],
                                                     "rgb_pixel_sha256": record["image_pixel_sha256"]})
        rows = rows + list(ima_rows.values())
        ima_info.update(state="completed_for_audited_images", unique_images=len(ima_rows), manifest_sha256=sha256(ima_path),
                        patient_lesion_scope="P1/PH2 patient metadata not provided by Task1 archives; image overlap only, no complete patient independence claim")
    summaries = {}
    for field in ["image_id", "rgb_sha256", "rgb_pixel_sha256", "mask_sha256", "mask_pixel_sha256"]:
        grouped = {}
        for row in rows:
            if row.get(field):
                grouped.setdefault(row[field], []).append(row)
        pairs = []
        summary = {}
        for value, group in grouped.items():
            for left, right in itertools.combinations(group, 2):
                group_left = f"{left['dataset']}:{left['split']}"
                group_right = f"{right['dataset']}:{right['split']}"
                if group_left == group_right:
                    continue
                key = " | ".join(sorted([group_left, group_right]))
                summary[key] = summary.get(key, 0) + 1
                pairs.append({"match_field": field, "match_value": value, "dataset_a": left["dataset"], "split_a": left["split"],
                              "image_id_a": left["image_id"], "dataset_b": right["dataset"], "split_b": right["split"], "image_id_b": right["image_id"]})
        fields = ["match_field", "match_value", "dataset_a", "split_a", "image_id_a", "dataset_b", "split_b", "image_id_b"]
        write_csv(PROJECT / f"data_manifests/external_overlap_by_{field}.csv", pairs, fields)
        summaries[field] = summary
    p1_only = {field: {key: count for key, count in matches.items() if "IMA" not in key and "PH2" not in key}
               for field, matches in summaries.items()}
    status["overlap_audit"] = {"ISIC2017_ISIC2018": p1_only, "all_acquired_protocols": summaries,
                               "PH2": "included in comparisons if acquired and fully decoded; otherwise unresolved",
                               "IMA": ima_info,
                               "note": "Mask-hash match alone is not evidence of duplicate RGB. ID, file-RGB and decoded-RGB matches are reported separately."}
    write_json(PROJECT / "data_manifests/external_overlap_summary.json", status["overlap_audit"])
    save_status(status)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/home/featurize/rsi_data/benchmarks"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--refresh-overlap-only", action="store_true")
    args = parser.parse_args()
    (PROJECT / "data_manifests").mkdir(parents=True, exist_ok=True)
    if args.refresh_overlap_only:
        status = json.loads((PROJECT / "outputs/external_data_status.json").read_text())
        rows = []
        for path in sorted((PROJECT / "data_manifests").glob("external_*_manifest.csv")):
            rows.extend(csv.DictReader(path.open()))
        overlap_audit(rows, status)
        return
    status = {"started_utc": utc(), "root": str(args.root), "sources": [], "datasets": {},
              "test_model_scoring_performed": False, "state": "working", "failure_summary": []}
    previous_status = PROJECT / "outputs/external_data_status.json"
    if args.audit_only and previous_status.exists():
        status = json.loads(previous_status.read_text())
        status["state"] = "auditing"
        jobs = status["sources"]
    else:
        jobs = discover_sources(status)
    save_status(status)
    if not args.audit_only:
        # Complete all small masks first. Large image archives then share four
        # connections. PH2 only follows links supplied by its official page.
        groups = [[job for job in jobs if job["kind"] == "masks"],
                  [job for job in jobs if job["kind"] == "images"] +
                  [job for job in jobs if job["dataset"] == "PH2"]]
        for group in groups:
            with ThreadPoolExecutor(args.workers) as pool:
                futures = {pool.submit(download, job, args.root, status): job for job in group}
                for future in as_completed(futures):
                    job = futures[future]
                    try:
                        archive = future.result()
                        if archive.suffix == ".zip":
                            extract_zip(job, archive, args.root, status)
                        else:
                            extract_ph2(archive, args.root, job, status)
                    except Exception as exc:
                        job.update(state="failed", terminal_error=repr(exc))
                        status["failure_summary"].append({"filename": job["filename"], "error": repr(exc)})
                        save_status(status)
    rows = []
    for dataset in EXPECTED:
        rows.extend(audit_dataset(dataset, args.root, status, args.workers))
    overlap_audit(rows, status)
    status["state"] = "complete" if all(item["all_files_passed"] for item in status["datasets"].values()) else "partial"
    status["finished_utc"] = utc()
    save_status(status)
    print(json.dumps({"state": status["state"], "datasets": status["datasets"], "failures": status["failure_summary"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
