import os
import re
import json
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from google.auth.transport.requests import Request as GoogleAuthRequest
from google.oauth2 import service_account
from requests.adapters import HTTPAdapter

# Use Railway's persistent volume when available; fall back to the local folder for PC testing.
DATA_DIR = Path(os.getenv("DATA_DIR", "/var/data"))
CACHE_DIR = DATA_DIR / "cache"
DB_FILE = DATA_DIR / "progress.db"
LOCAL_SERVICE_ACCOUNT_FILE = Path(__file__).resolve().parent / "service-account.json"

# Each cached piece is 8 MiB; this keeps seeking reasonably responsive without huge downloads.
CHUNK_SIZE = 8 * 1024 * 1024

# Limit total cached video data on the persistent volume to 5 GiB.
MAX_CACHE_TOTAL = 5 * 1024 * 1024 * 1024

# Prefetch the next two chunks after the currently requested chunk.
PREFETCH_CHUNKS = 2

# Limit background prefetch concurrency so the server does not overload Google Drive.
PREFETCH_WORKERS = 3

# Keep Drive metadata in memory for five minutes.
METADATA_TTL = 300

# Google Drive folder is shared with the service account.
VIDEOS = {
    "E1": {"name": "E1.mp4", "id": "1pu4q_blEkl90Xz8WVuocHdzPOD07pvo2"},
    "E2": {"name": "E2.mp4", "id": "1rbBr7CoYsbFMn7ikTXt6LyrUXtY_KVvB"},
    "E3": {"name": "E3.mp4", "id": "1Mo-QxK5kgAcqQbTCEibKyqN4qUQOWfgO"},
    "E4": {"name": "E4.mp4", "id": "1AWdLM0k6Bv9VvWkLEDV-EWBPgsbWxOsn"},
    "E5": {"name": "E5.mp4", "id": "1tV0EH3s7PQgOZDp5e5yCKF3xhIcv4g7q"},
    "E6": {"name": "E6.mp4", "id": "1r0QVmacbJW4KAGJzJKvwac-U4R_V9QTr"},
    "E7": {"name": "E7.mp4", "id": "1dBLq2QG9lPpisesfhQ5MlGoJpQocE47d"},
    "E8": {"name": "E8.mp4", "id": "1v7EV6EfgTPnnzP2Bkt81bn_1dS1-FGRW"},
}

# Create the persistent directories before the application starts.
CACHE_DIR.mkdir(parents=True, exist_ok=True)
DB_FILE.parent.mkdir(parents=True, exist_ok=True)

# Railway stores the Google service-account JSON in an environment variable.
service_account_json = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")

if service_account_json:
    try:
        service_account_info = json.loads(service_account_json)
    except json.JSONDecodeError as exc:
        raise RuntimeError("GOOGLE_SERVICE_ACCOUNT_JSON is not valid JSON") from exc

    # Build credentials directly from the Railway environment variable.
    credentials = service_account.Credentials.from_service_account_info(
        service_account_info,
        scopes=["https://www.googleapis.com/auth/drive.readonly"],
    )
else:
    # Keep local testing compatible with service-account.json beside server.py.
    if not LOCAL_SERVICE_ACCOUNT_FILE.exists():
        raise RuntimeError(
            "Google credentials are missing. Set GOOGLE_SERVICE_ACCOUNT_JSON on Railway "
            "or place service-account.json beside server.py for local testing."
        )

    # Load the local JSON key only when the environment variable is not present.
    credentials = service_account.Credentials.from_service_account_file(
        str(LOCAL_SERVICE_ACCOUNT_FILE),
        scopes=["https://www.googleapis.com/auth/drive.readonly"],
    )

# Protect credential refresh because several requests may arrive at the same time.
credentials_lock = threading.Lock()

# Give each worker thread its own reusable HTTP session and connection pool.
thread_local = threading.local()

# In-memory metadata cache and its lock.
metadata_cache = {}
metadata_lock = threading.Lock()

# Locks prevent two requests from downloading the same cache chunk simultaneously.
chunk_locks = {}
chunk_locks_guard = threading.Lock()

# Background workers are used for prefetching upcoming chunks.
executor = ThreadPoolExecutor(max_workers=PREFETCH_WORKERS)

# Create the FastAPI application.
app = FastAPI(title="Drive Video Streaming")


def get_session():
    # Reuse one HTTP connection pool per Python worker thread.
    if not hasattr(thread_local, "session"):
        session = requests.Session()
        adapter = HTTPAdapter(
            pool_connections=20,
            pool_maxsize=20,
            max_retries=2,
        )
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        thread_local.session = session
    return thread_local.session


def get_access_token():
    # Refresh the Google token only when necessary.
    with credentials_lock:
        if not credentials.valid or credentials.expired:
            credentials.refresh(GoogleAuthRequest())
        return credentials.token


def drive_headers(extra=None):
    # Build authenticated headers for Google Drive API requests.
    headers = {"Authorization": "Bearer " + get_access_token()}
    if extra:
        headers.update(extra)
    return headers


def get_video(video_key):
    # Normalize the requested video key to uppercase.
    key = video_key.upper()
    video = VIDEOS.get(key)

    # Return 404 for unknown video names instead of exposing arbitrary Drive IDs.
    if video is None:
        raise HTTPException(status_code=404, detail="Video not found")

    return key, video


def get_metadata(file_id):
    # Use cached metadata when it is still fresh.
    now = time.time()
    with metadata_lock:
        cached = metadata_cache.get(file_id)
        if cached and now - cached["time"] < METADATA_TTL:
            return cached["data"]

    # Ask Drive only for the metadata required by the streaming layer.
    url = "https://www.googleapis.com/drive/v3/files/" + file_id
    response = get_session().get(
        url,
        params={"fields": "id,name,size,mimeType"},
        headers=drive_headers(),
        timeout=(10, 30),
    )

    if response.status_code != 200:
        response.close()
        raise HTTPException(status_code=502, detail="Google Drive metadata error")

    data = response.json()
    response.close()

    # Google Drive must provide the file size for byte-range streaming.
    if "size" not in data:
        raise HTTPException(status_code=502, detail="Google Drive did not return file size")

    data["size"] = int(data["size"])

    # Store metadata in the in-memory cache.
    with metadata_lock:
        metadata_cache[file_id] = {"time": now, "data": data}

    return data


def get_chunk_lock(file_id, index):
    # Identify a cache lock by Google Drive file ID and chunk index.
    key = (file_id, index)

    # Create the lock once and reuse it for subsequent requests.
    with chunk_locks_guard:
        if key not in chunk_locks:
            chunk_locks[key] = threading.Lock()
        return chunk_locks[key]


def cache_dir(file_id):
    # Keep chunks of each Drive video in a separate directory.
    path = CACHE_DIR / file_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def chunk_path(file_id, index):
    # Use fixed-width chunk names to keep directory listings predictable.
    return cache_dir(file_id) / ("chunk_%08d.bin" % index)


def evict_cache():
    # Scan cached chunks and calculate their total size.
    files = []
    total = 0

    for path in CACHE_DIR.rglob("chunk_*.bin"):
        try:
            stat = path.stat()
            total += stat.st_size
            files.append((stat.st_atime, path, stat.st_size))
        except OSError:
            pass

    # Nothing needs to be removed when the cache is below its configured limit.
    if total <= MAX_CACHE_TOTAL:
        return

    # Delete least-recently-accessed chunks first.
    files.sort(key=lambda item: item[0])

    for _, path, size in files:
        if total <= MAX_CACHE_TOTAL:
            break
        try:
            path.unlink()
            total -= size
        except OSError:
            pass


def download_chunk(file_id, index, file_size):
    # Calculate the exact byte range represented by this cache chunk.
    start = index * CHUNK_SIZE

    if start >= file_size:
        return None

    end = min(start + CHUNK_SIZE - 1, file_size - 1)
    expected = end - start + 1
    path = chunk_path(file_id, index)

    # Only one request at a time may populate a given cache chunk.
    with get_chunk_lock(file_id, index):
        try:
            # Reuse a complete cached chunk instead of contacting Google Drive.
            if path.exists() and path.stat().st_size == expected:
                os.utime(path, None)
                return path
        except OSError:
            pass

        # Request exactly the needed bytes from Google Drive.
        url = "https://www.googleapis.com/drive/v3/files/" + file_id
        response = get_session().get(
            url,
            params={"alt": "media"},
            headers=drive_headers({"Range": "bytes=%d-%d" % (start, end)}),
            stream=True,
            timeout=(10, 120),
        )

        # Drive must honor the byte-range request.
        if response.status_code != 206:
            response.close()
            raise HTTPException(status_code=502, detail="Google Drive range request failed")

        # Validate the returned byte range before writing it to disk.
        content_range = response.headers.get("Content-Range", "")
        match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+|\*)", content_range)

        if not match or int(match.group(1)) != start or int(match.group(2)) != end:
            response.close()
            raise HTTPException(status_code=502, detail="Unexpected Google Drive byte range")

        # Write to a temporary file first so an interrupted download cannot corrupt the cache.
        temp = path.with_suffix(".tmp")
        written = 0

        try:
            with open(temp, "wb") as output:
                for block in response.iter_content(chunk_size=1024 * 1024):
                    if block:
                        output.write(block)
                        written += len(block)

            # Reject incomplete chunks.
            if written != expected:
                raise HTTPException(status_code=502, detail="Incomplete chunk from Google Drive")

            # Atomically replace the old file with the complete chunk.
            os.replace(temp, path)

            # Enforce the global cache limit after a successful download.
            evict_cache()

            return path
        finally:
            response.close()
            try:
                temp.unlink(missing_ok=True)
            except OSError:
                pass


def prefetch(file_id, first_index, file_size):
    # Calculate the last valid chunk index for this file.
    last_index = (file_size - 1) // CHUNK_SIZE

    # Download the next few chunks in the background.
    for offset in range(1, PREFETCH_CHUNKS + 1):
        index = first_index + offset

        if index > last_index:
            break

        try:
            download_chunk(file_id, index, file_size)
        except Exception:
            # Prefetch failure must not interrupt the current video request.
            pass


def iter_range(file_id, start, end, file_size):
    # Determine which cache chunks overlap the requested HTTP range.
    first = start // CHUNK_SIZE
    last = end // CHUNK_SIZE

    for index in range(first, last + 1):
        # Download or reuse the required cache chunk.
        path = download_chunk(file_id, index, file_size)

        if path is None:
            return

        # Calculate the byte offsets inside this particular chunk.
        chunk_start = index * CHUNK_SIZE
        offset = max(start, chunk_start) - chunk_start
        limit = min(end, chunk_start + CHUNK_SIZE - 1) - chunk_start

        # Stream the requested portion in 1 MiB blocks.
        with open(path, "rb") as source:
            source.seek(offset)
            remaining = limit - offset + 1

            while remaining > 0:
                block = source.read(min(1024 * 1024, remaining))

                if not block:
                    return

                remaining -= len(block)

                # Update access time so frequently used chunks are kept longer.
                try:
                    os.utime(path, None)
                except OSError:
                    pass

                yield block


def parse_range(header, file_size):
    # No Range header means return the whole file.
    if not header:
        return 0, file_size - 1

    # Support the standard single-range form used by HTML5 video players.
    match = re.fullmatch(r"bytes=(\d*)-(\d*)", header.strip())

    if not match:
        raise HTTPException(status_code=416, detail="Invalid Range header")

    start_text, end_text = match.groups()

    if not start_text and not end_text:
        raise HTTPException(status_code=416, detail="Invalid Range header")

    # Handle suffix ranges such as bytes=-500000.
    if not start_text:
        length = int(end_text)

        if length <= 0:
            raise HTTPException(status_code=416, detail="Invalid Range header")

        start = max(file_size - length, 0)
        end = file_size - 1

    else:
        # Handle ranges such as bytes=100000- or bytes=100000-200000.
        start = int(start_text)
        end = int(end_text) if end_text else file_size - 1

        if start >= file_size or start > end:
            raise HTTPException(status_code=416, detail="Range not satisfiable")

        end = min(end, file_size - 1)

    return start, end


def init_db():
    # Open the persistent SQLite database.
    connection = sqlite3.connect(DB_FILE, timeout=30)

    try:
        # Create the progress table if this is the first deployment.
        connection.execute(
            "CREATE TABLE IF NOT EXISTS progress ("
            "viewer_id TEXT NOT NULL, "
            "video_key TEXT NOT NULL, "
            "position REAL NOT NULL DEFAULT 0, "
            "duration REAL NOT NULL DEFAULT 0, "
            "updated_at INTEGER NOT NULL, "
            "PRIMARY KEY (viewer_id, video_key)"
            ")"
        )
        connection.commit()
    finally:
        connection.close()


# Initialize SQLite before serving requests.
init_db()


@app.get("/health")
def health():
    # Simple endpoint for Railway health checks and manual testing.
    return {
        "status": "ok",
        "videos": len(VIDEOS),
        "chunk_size_mb": CHUNK_SIZE // 1024 // 1024,
    }


@app.get("/api/videos")
def api_videos():
    # Return the fixed public video list without exposing Drive file IDs.
    return [{"key": key, "name": video["name"]} for key, video in VIDEOS.items()]


@app.get("/api/progress/{video_key}")
def get_progress(video_key: str, viewer_id: str = ""):
    # Validate the requested video key.
    key, _ = get_video(video_key)

    # Limit the localStorage-provided viewer ID to a safe size.
    viewer_id = viewer_id.strip()[:128]

    if not viewer_id:
        return {"position": 0, "duration": 0}

    # Read the saved position from persistent SQLite storage.
    connection = sqlite3.connect(DB_FILE, timeout=30)

    try:
        row = connection.execute(
            "SELECT position, duration FROM progress WHERE viewer_id=? AND video_key=?",
            (viewer_id, key),
        ).fetchone()
    finally:
        connection.close()

    if row is None:
        return {"position": 0, "duration": 0}

    return {"position": row[0], "duration": row[1]}


@app.post("/api/progress/{video_key}")
async def save_progress(video_key: str, request: Request):
    # Validate the requested video key.
    key, _ = get_video(video_key)

    try:
        # Parse the JSON body sent by the browser.
        data = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    # Read and limit the browser-specific viewer identifier.
    viewer_id = str(data.get("viewer_id", "")).strip()[:128]

    if not viewer_id:
        raise HTTPException(status_code=400, detail="viewer_id is required")

    try:
        # Normalize playback position and duration to non-negative floats.
        position = max(0.0, float(data.get("position", 0)))
        duration = max(0.0, float(data.get("duration", 0)))
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="Invalid position or duration")

    # Treat a video within two seconds of the end as completed.
    if duration > 0 and position >= max(duration - 2, 0):
        position = 0.0

    # Store or update the viewer's playback position.
    connection = sqlite3.connect(DB_FILE, timeout=30)

    try:
        connection.execute(
            "INSERT INTO progress (viewer_id, video_key, position, duration, updated_at) "
            "VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(viewer_id, video_key) DO UPDATE SET "
            "position=excluded.position, duration=excluded.duration, updated_at=excluded.updated_at",
            (viewer_id, key, position, duration, int(time.time())),
        )
        connection.commit()
    finally:
        connection.close()

    return {"ok": True}


@app.get("/video/{video_key}")
def stream_video(video_key: str, request: Request):
    # Resolve the public video key to its private Drive ID.
    _, video = get_video(video_key)

    # Read the Drive file size, using the metadata cache when possible.
    metadata = get_metadata(video["id"])
    file_size = metadata["size"]

    # Convert the browser's Range header into concrete byte offsets.
    start, end = parse_range(request.headers.get("range"), file_size)

    # Return 206 for range requests and 200 for a normal full-file request.
    status = 206 if request.headers.get("range") else 200

    # Start downloading the next chunks in the background.
    executor.submit(prefetch, video["id"], start // CHUNK_SIZE, file_size)

    # Tell browsers and HTML5 players that byte ranges are supported.
    headers = {
        "Accept-Ranges": "bytes",
        "Content-Length": str(end - start + 1),
        "Content-Type": "video/mp4",
        "Cache-Control": "public, max-age=3600",
        "X-Accel-Buffering": "no",
    }

    # Add the mandatory Content-Range header for HTTP 206 responses.
    if status == 206:
        headers["Content-Range"] = "bytes %d-%d/%d" % (start, end, file_size)

    # Stream only the requested range to the browser.
    return StreamingResponse(
        iter_range(video["id"], start, end, file_size),
        status_code=status,
        headers=headers,
        media_type="video/mp4",
    )


@app.head("/video/{video_key}")
def head_video(video_key: str):
    # Resolve the requested public video key.
    _, video = get_video(video_key)

    # Read the current file size from Google Drive metadata.
    metadata = get_metadata(video["id"])

    # Return headers only, which is what a browser needs for a HEAD request.
    return Response(
        status_code=200,
        headers={
            "Accept-Ranges": "bytes",
            "Content-Length": str(metadata["size"]),
            "Content-Type": "video/mp4",
        },
    )


PAGE_TEMPLATE = """<!doctype html>
<html lang="vi">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<style>
body { font-family: Arial,sans-serif; background:#111; color:#eee; margin:0; padding:20px; }
.wrap { max-width:1100px; margin:auto; }
h1 { font-size:22px; margin:0 0 15px; }
video { width:100%; max-height:75vh; background:#000; border-radius:8px; }
.list { display:flex; gap:8px; flex-wrap:wrap; margin-top:15px; }
.item { padding:9px 14px; background:#292929; color:#eee; text-decoration:none; border-radius:6px; }
.active { background:#555; }
.info { margin-top:10px; color:#aaa; font-size:14px; }
</style>
</head>
<body>
<div class="wrap">
<h1>__TITLE__</h1>
<video id="player" controls playsinline preload="metadata"></video>
<div class="list">__BUTTONS__</div>
<div class="info" id="info">Đang tải vị trí xem...</div>
</div>
<script>
const VIDEO_KEY = "__VIDEO_KEY__";
const player = document.getElementById("player");
const info = document.getElementById("info");
const viewerStorageKey = "drive_stream_viewer_id";
const positionStorageKey = "drive_stream_position_" + VIDEO_KEY;
let viewerId = localStorage.getItem(viewerStorageKey);

if (!viewerId) {
    viewerId = (crypto.randomUUID ? crypto.randomUUID() : String(Date.now()) + Math.random());
    localStorage.setItem(viewerStorageKey, viewerId);
}

player.src = "/video/" + VIDEO_KEY;
let lastSaved = 0;

async function loadProgress() {
    try {
        const response = await fetch(
            "/api/progress/" + VIDEO_KEY + "?viewer_id=" + encodeURIComponent(viewerId)
        );
        const data = await response.json();
        const localPosition = parseFloat(localStorage.getItem(positionStorageKey) || "0");
        const position = Math.max(Number(data.position || 0), localPosition);

        player.addEventListener("loadedmetadata", function() {
            if (position > 5 && position < player.duration - 5) {
                player.currentTime = position;
                info.textContent = "Tiếp tục từ " + formatTime(position);
            } else {
                info.textContent = "Video mới";
            }
        }, {once:true});
    } catch (error) {
        info.textContent = "Không lấy được vị trí xem cũ";
    }
}

async function saveProgress(force) {
    if (!Number.isFinite(player.currentTime)) return;

    const position = player.currentTime;
    const duration = Number.isFinite(player.duration) ? player.duration : 0;
    localStorage.setItem(positionStorageKey, String(position));

    const now = Date.now();
    if (!force && now - lastSaved < 1500) return;
    lastSaved = now;

    const payload = JSON.stringify({
        viewer_id: viewerId,
        position: position,
        duration: duration
    });

    try {
        await fetch("/api/progress/" + VIDEO_KEY, {
            method:"POST",
            headers:{"Content-Type":"application/json"},
            body:payload,
            keepalive:true
        });
    } catch (error) {}
}

function formatTime(seconds) {
    seconds = Math.floor(seconds);
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    const s = seconds % 60;
    return (h ? h + ":" : "") + String(m).padStart(2,"0") + ":" + String(s).padStart(2,"0");
}

player.addEventListener("timeupdate", function() {
    saveProgress(false);
});

player.addEventListener("pause", function() {
    saveProgress(true);
});

player.addEventListener("ended", function() {
    localStorage.removeItem(positionStorageKey);
    fetch("/api/progress/" + VIDEO_KEY, {
        method:"POST",
        headers:{"Content-Type":"application/json"},
        body:JSON.stringify({
            viewer_id:viewerId,
            position:0,
            duration:player.duration
        }),
        keepalive:true
    }).catch(function() {});
});

window.addEventListener("beforeunload", function() {
    if (!Number.isFinite(player.currentTime)) return;

    const payload = JSON.stringify({
        viewer_id:viewerId,
        position:player.currentTime || 0,
        duration:player.duration || 0
    });

    navigator.sendBeacon(
        "/api/progress/" + VIDEO_KEY,
        new Blob([payload], {type:"application/json"})
    );
});

loadProgress();
</script>
</body>
</html>"""


@app.get("/watch/{video_key}", response_class=HTMLResponse)
def watch_page(video_key: str):
    # Resolve the requested video and build the navigation buttons.
    key, video = get_video(video_key)

    buttons = "".join(
        '<a class="item%s" href="/watch/%s">%s</a>'
        % (" active" if item_key == key else "", item_key, item["name"])
        for item_key, item in VIDEOS.items()
    )

    # Replace placeholders without using Python f-strings, avoiding JavaScript brace errors.
    html = PAGE_TEMPLATE.replace("__TITLE__", video["name"])
    html = html.replace("__VIDEO_KEY__", key)
    html = html.replace("__BUTTONS__", buttons)

    return HTMLResponse(html)


@app.get("/", response_class=HTMLResponse)
def index():
    # Build a simple homepage containing links to all videos.
    items = "".join(
        '<li><a href="/watch/%s">%s</a></li>' % (key, video["name"])
        for key, video in VIDEOS.items()
    )

    return HTMLResponse(
        "<!doctype html><html lang='vi'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>Videos</title></head><body>"
        "<h1>Videos</h1><ul>" + items + "</ul>"
        "<p><a href='/health'>Server health</a></p>"
        "</body></html>"
    )
