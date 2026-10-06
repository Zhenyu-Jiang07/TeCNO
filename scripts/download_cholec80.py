"""Download the official archive with resumable, validated HTTP range requests.

No credentials needed. Each fsynced chunk has a SHA256 completion marker.
The archive is sparse while incomplete; use markers, not file size, for progress.
"""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.request

URL = "https://s3.unistra.fr/camma_public/datasets/cholec80/cholec80.zip"
ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data/raw/cholec80.zip"
CHUNK = 32 * 1024**2


def main():
    DEST.parent.mkdir(parents=True, exist_ok=True)
    markers = DEST.with_suffix(".chunks")
    markers.mkdir(exist_ok=True)
    with urllib.request.urlopen(urllib.request.Request(URL, method="HEAD"), timeout=60) as r:
        size = int(r.headers["Content-Length"])
        identity = {"url": URL, "bytes": size, "etag": r.headers["ETag"],
                    "last_modified": r.headers["Last-Modified"], "chunk_bytes": CHUNK,
                    "license": "CC-BY-NC-SA-4.0", "entry": "https://camma.unistra.fr/datasets/"}
    meta = markers / "source.json"
    if meta.exists():
        assert json.loads(meta.read_text()) == identity, "Remote archive changed; do not mix versions"
    else:
        meta.write_text(json.dumps(identity, indent=2))
    fd = os.open(DEST, os.O_CREAT | os.O_RDWR, 0o644)
    os.ftruncate(fd, size)
    total = (size + CHUNK - 1) // CHUNK

    def fetch(i):
        start, end = i * CHUNK, min(size, (i + 1) * CHUNK) - 1
        done = markers / f"{i:05d}.json"
        if done.exists():
            expected = json.loads(done.read_text())["sha256"]
            if hashlib.sha256(os.pread(fd, end-start+1, start)).hexdigest() == expected:
                return False
        for attempt in range(10):
            try:
                req = urllib.request.Request(URL, headers={"Range": f"bytes={start}-{end}",
                                                           "If-Match": identity["etag"]})
                with urllib.request.urlopen(req, timeout=120) as r:
                    assert r.status == 206, f"Server ignored Range: {r.status}"
                    assert r.headers["Content-Range"] == f"bytes {start}-{end}/{size}"
                    blob = r.read()
                assert len(blob) == end-start+1
                offset = 0
                while offset < len(blob):
                    offset += os.pwrite(fd, blob[offset:], start + offset)
                os.fsync(fd)
                tmp = done.with_suffix(".tmp")
                tmp.write_text(json.dumps({"bytes": len(blob), "sha256": hashlib.sha256(blob).hexdigest()}))
                tmp.replace(done)
                return True
            except Exception as exc:
                print(f"retry chunk={i} attempt={attempt+1}: {exc}", flush=True)
                time.sleep(min(30, 2**attempt))
        raise RuntimeError(f"chunk {i} failed")

    started = time.monotonic()
    count = 0
    # Fetch annotations + central directory first so preparation can overlap transfer.
    order = list(range(max(0, total-4), total))
    index_path = ROOT / 'sources/cholec80-archive-index.json'
    if index_path.exists():
        index = {x['name']: x for x in json.loads(index_path.read_text())}
        # First real-data smoke cases, then training/validation, then test cases.
        for video in [36,41] + list(range(1,81)):
            item = index[f'videos/video{video:02d}.mp4']
            order.extend(range(item['offset']//CHUNK,
                               (item['offset']+item['compressed']+256)//CHUNK+1))
    order = list(dict.fromkeys(order + list(range(total))))
    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as pool:
        for future in concurrent.futures.as_completed([pool.submit(fetch, i) for i in order]):
            future.result()
            count += 1
            if count % 16 == 0 or count == total:
                print(f"chunks={count}/{total} progress={100*count/total:.2f}% elapsed_s={time.monotonic()-started:.0f}", flush=True)
    os.close(fd)
    digest = hashlib.file_digest(open(DEST, "rb"), "sha256").hexdigest()
    (DEST.parent / "cholec80.zip.sha256").write_text(f"{digest}  cholec80.zip\n")
    (markers / "COMPLETE").write_text(digest + "\n")
    print(f"COMPLETE sha256={digest}", flush=True)


if __name__ == "__main__":
    main()
