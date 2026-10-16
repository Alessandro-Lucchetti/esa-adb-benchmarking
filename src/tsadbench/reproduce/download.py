"""HTTP helpers of the reproduction: resumable downloads checked against Zenodo's md5, and
read access to single members of a large remote zip through range requests."""
import hashlib
import http.cookiejar
import io
import json
import shutil
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

CHUNK = 8 << 20
RETRIES = 5


def opener() -> urllib.request.OpenerDirector:
    """URL opener with a cookie jar (the ESA-ADB SharePoint links need the session cookie
    set by the first visit)."""
    op = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
    op.addheaders = [("User-Agent", "esa-adb-benchmarking-reproduce")]
    return op


def _open(op: urllib.request.OpenerDirector, url: str, headers: dict | None = None):
    for attempt in range(RETRIES):
        try:
            return op.open(urllib.request.Request(url, headers=headers or {}), timeout=120)
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt == RETRIES - 1:
                raise
            time.sleep(2 ** (attempt + 1))


def zenodo_file(record: str | int, key: str) -> tuple[str, str, int]:
    """Download URL, md5 and size of file ``key`` of a Zenodo record."""
    with _open(opener(), f"https://zenodo.org/api/records/{record}") as r:
        meta = json.load(r)
    for f in meta["files"]:
        if f["key"] == key:
            return f"https://zenodo.org/records/{record}/files/{key}?download=1", \
                f["checksum"].removeprefix("md5:"), int(f["size"])
    raise FileNotFoundError(f"{key} not in Zenodo record {record}")


def md5sum(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while block := f.read(CHUNK):
            h.update(block)
    return h.hexdigest()


def download(url: str, dest: Path, md5: str | None = None, size: int | None = None) -> Path:
    """Download ``url`` to ``dest``, resuming a partial ``dest.part``; checks size and md5."""
    if dest.exists() and (md5 is None or md5sum(dest) == md5):
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    op = opener()
    tty = sys.stdout.isatty()
    while True:
        have = part.stat().st_size if part.exists() else 0
        if size is not None and have >= size:
            break
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            with _open(op, url, headers) as r, open(part, "ab" if have and r.status == 206 else "wb") as f:
                total = size or int(r.headers.get("Content-Length", 0)) + (have if r.status == 206 else 0)
                done, last = (have if r.status == 206 else 0), time.time()
                while block := r.read(CHUNK):
                    f.write(block)
                    done += len(block)
                    if not total:
                        continue
                    msg = f"  {dest.name}: {done / 2**20:,.0f} / {total / 2**20:,.0f} MB"
                    if tty:
                        print("\r" + msg, end="", flush=True)
                    elif time.time() - last >= 30:   # a log (docker logs) shows whole lines only
                        print(msg, flush=True)
                        last = time.time()
            if tty:
                print()
            if size is None or part.stat().st_size >= size:
                break
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            print(f"\n  interrupted ({e}), resuming", file=sys.stderr)
            time.sleep(5)
    if size is not None and part.stat().st_size != size:
        raise OSError(f"{dest.name}: {part.stat().st_size} bytes, expected {size}")
    if md5 is not None and (got := md5sum(part)) != md5:
        part.unlink()
        raise OSError(f"{dest.name}: md5 {got}, expected {md5}; partial file removed, run again")
    part.replace(dest)
    return dest


class HTTPRangeFile(io.RawIOBase):
    """Seekable, read-only view of a remote file through HTTP range requests, enough for
    ``zipfile`` to read single members of a remote archive without downloading it."""

    def __init__(self, url: str, op: urllib.request.OpenerDirector | None = None):
        self.op = op or opener()
        with _open(self.op, url, {"Range": "bytes=0-0"}) as r:
            if r.status != 206:
                raise OSError(f"{url}: the server does not honour range requests")
            self.url = r.geturl()
            self.size = int(r.headers["Content-Range"].rsplit("/", 1)[1])
        self.pos = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        self.pos = {io.SEEK_SET: offset, io.SEEK_CUR: self.pos + offset, io.SEEK_END: self.size + offset}[whence]
        return self.pos

    def readinto(self, b) -> int:
        if self.pos >= self.size:
            return 0
        end = min(self.pos + len(b), self.size) - 1
        with _open(self.op, self.url, {"Range": f"bytes={self.pos}-{end}"}) as r:
            data = r.read()
        b[:len(data)] = data
        self.pos += len(data)
        return len(data)


def remote_zip(url: str, op: urllib.request.OpenerDirector | None = None) -> zipfile.ZipFile:
    return zipfile.ZipFile(io.BufferedReader(HTTPRangeFile(url, op), buffer_size=CHUNK))


def extract_members(z: zipfile.ZipFile, names: dict[str, Path]) -> None:
    """Copy archive members to the given paths (zipfile checks each CRC on the way)."""
    for name, dest in names.items():
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + ".part")
        with z.open(name) as src, open(tmp, "wb") as f:
            shutil.copyfileobj(src, f, CHUNK)
        tmp.replace(dest)
