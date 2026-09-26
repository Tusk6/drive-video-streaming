import os
import re
import json
import hashlib
import html
import shutil
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

# Use Railway's persistent volume when available, and use a local folder when testing on the PC.
DATA_DIR = Path(os.getenv("DATA_DIR", "/var/data"))
# Store downloaded video chunks inside the persistent data directory.
CACHE_DIR = DATA_DIR / "cache"
# Store playback positions in SQLite on the persistent volume.
DB_FILE = DATA_DIR / "progress.db"
# Keep local testing compatible with the service-account.json file beside server.py.
LOCAL_SERVICE_ACCOUNT_FILE = Path(__file__).resolve().parent / "service-account.json"

# This is the Google Drive folder ID that contains standalone movies and series folders.
DRIVE_ROOT_FOLDER_ID = os.getenv("DRIVE_ROOT_FOLDER_ID", "1ON4DZCGmOV_Uka65UD4tMqEIQOdNygyb")
# Cache each video in 8 MiB pieces to keep seeking reasonably responsive.
CHUNK_SIZE = 8 * 1024 * 1024
# Keep the persistent video cache capped at 2 GiB to avoid filling the Railway volume.
MAX_CACHE_TOTAL = 2 * 1024 * 1024 * 1024
# Prefetch the next two pieces while the current piece is being watched.
PREFETCH_CHUNKS = 2
# Limit background prefetch concurrency so Google Drive is not unnecessarily hammered.
PREFETCH_WORKERS = 3
# Refresh the Google Drive library at most once per minute during normal browsing.
LIBRARY_TTL = 60
# Keep individual Drive file metadata in memory for five minutes.
METADATA_TTL = 300
# Only these video extensions are exposed by the streaming library.
VIDEO_EXTENSIONS = {".mp4", ".m4v", ".webm", ".mov"}
# The service can optionally protect the cache-clear endpoint with a Railway variable.
CACHE_ADMIN_TOKEN = os.getenv("CACHE_ADMIN_TOKEN", "").strip()

# Create persistent directories before the application starts.
CACHE_DIR.mkdir(parents=True, exist_ok=True)
DB_FILE.parent.mkdir(parents=True, exist_ok=True)

# Read the Google service-account JSON from Railway when deployed.
service_account_json = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")

# Build Google credentials from the Railway environment variable when it exists.
if service_account_json:
    try:
        # Parse the JSON string stored in the Railway variable.
        service_account_info = json.loads(service_account_json)
    except json.JSONDecodeError as exc:
        # Stop startup with a clear message if the Railway variable is malformed.
        raise RuntimeError("GOOGLE_SERVICE_ACCOUNT_JSON is not valid JSON") from exc
    # Create read-only Drive credentials without writing the secret to disk.
    credentials = service_account.Credentials.from_service_account_info(
        service_account_info,
        scopes=["https://www.googleapis.com/auth/drive.readonly"],
    )
else:
    # Fall back to the local service-account.json for PC testing.
    if not LOCAL_SERVICE_ACCOUNT_FILE.exists():
        # Explain exactly how local credentials should be supplied.
        raise RuntimeError(
            "Google credentials are missing. Set GOOGLE_SERVICE_ACCOUNT_JSON on Railway "
            "or place service-account.json beside server.py for local testing."
        )
    # Load the local JSON key without exposing its contents.
    credentials = service_account.Credentials.from_service_account_file(
        str(LOCAL_SERVICE_ACCOUNT_FILE),
        scopes=["https://www.googleapis.com/auth/drive.readonly"],
    )

# Protect Google token refresh because several requests can arrive simultaneously.
credentials_lock = threading.Lock()
# Give each Python worker thread its own reusable HTTP connection pool.
thread_local = threading.local()
# Keep Google Drive file metadata in memory for a short period.
metadata_cache = {}
# Protect the metadata cache from concurrent access.
metadata_lock = threading.Lock()
# Prevent duplicate downloads of the same file chunk.
chunk_locks = {}
# Protect the chunk-lock dictionary itself.
chunk_locks_guard = threading.Lock()
# Cache the discovered Drive library in memory.
library_cache = None
# Store the time at which the Drive library was last refreshed.
library_cache_time = 0.0
# Protect library refreshes so many users do not trigger simultaneous Drive listings.
library_lock = threading.Lock()
# Run prefetch jobs in the background.
executor = ThreadPoolExecutor(max_workers=PREFETCH_WORKERS)
# Create the FastAPI application.
app = FastAPI(title="Drive Video Streaming")


def get_session():
    # Reuse one HTTP connection pool per Python worker thread.
    if not hasattr(thread_local, "session"):
        # Create a persistent requests session for this thread.
        session = requests.Session()
        # Configure a reusable HTTPS connection pool with a small retry policy.
        adapter = HTTPAdapter(pool_connections=20, pool_maxsize=20, max_retries=2)
        # Attach the adapter to HTTPS requests.
        session.mount("https://", adapter)
        # Attach the adapter to HTTP requests as well.
        session.mount("http://", adapter)
        # Save the session on thread-local storage for later requests.
        thread_local.session = session
    # Return the current thread's reusable session.
    return thread_local.session


def get_access_token():
    # Protect token refresh so concurrent requests do not refresh the same token twice.
    with credentials_lock:
        # Refresh the Google token only when it is missing or expired.
        if not credentials.valid or credentials.expired:
            credentials.refresh(GoogleAuthRequest())
        # Return the valid bearer token.
        return credentials.token


def drive_headers(extra=None):
    # Build authenticated headers for Google Drive API requests.
    headers = {"Authorization": "Bearer " + get_access_token()}
    # Add request-specific headers such as Range when supplied.
    if extra:
        headers.update(extra)
    # Return the final header dictionary.
    return headers


def drive_list(query, fields):
    # Start with no page token because the Drive API may paginate large libraries.
    page_token = None
    # Accumulate every page of matching Drive items.
    items = []
    # Continue until Drive reports that there is no next page.
    while True:
        # Request one page from the Drive Files API.
        response = get_session().get(
            "https://www.googleapis.com/drive/v3/files",
            params={
                "q": query,
                "spaces": "drive",
                "pageSize": 1000,
                "orderBy": "name_natural",
                "fields": fields + ",nextPageToken",
                "pageToken": page_token,
            },
            headers=drive_headers(),
            timeout=(10, 30),
        )
        # Convert Google Drive failures into a server-side error.
        if response.status_code != 200:
            response.close()
            raise HTTPException(status_code=502, detail="Google Drive library listing failed")
        # Decode the current page.
        data = response.json()
        # Close the HTTP response as soon as its JSON has been consumed.
        response.close()
        # Append the current page's files and folders.
        items.extend(data.get("files", []))
        # Read the next page token, if one exists.
        page_token = data.get("nextPageToken")
        # Stop after the final page.
        if not page_token:
            break
    # Return the complete list of matching items.
    return items


def make_key(file_id):
    # Hash the private Drive ID so public URLs do not directly expose Drive IDs.
    digest = hashlib.sha256(file_id.encode("utf-8")).hexdigest()
    # A 24-character prefix is enough to make accidental collisions negligible here.
    return digest[:24]


def is_video_file(item):
    # Read the file name without trusting the MIME type alone.
    name = item.get("name", "")
    # Extract the lower-case extension.
    extension = Path(name).suffix.lower()
    # Accept only configured video extensions.
    return extension in VIDEO_EXTENSIONS


def build_library(force=False):
    # Use the cached library when it is still fresh and a forced refresh was not requested.
    global library_cache, library_cache_time
    now = time.time()
    if not force and library_cache is not None and now - library_cache_time < LIBRARY_TTL:
        return library_cache

    # Serialize refreshes so only one request lists Google Drive at a time.
    with library_lock:
        # Re-check the cache after waiting for another request to finish refreshing it.
        now = time.time()
        if not force and library_cache is not None and now - library_cache_time < LIBRARY_TTL:
            return library_cache

        # List video files directly inside the root folder as standalone movies.
        root_videos = drive_list(
            "'" + DRIVE_ROOT_FOLDER_ID + "' in parents and trashed = false and mimeType != 'application/vnd.google-apps.folder'",
            "files(id,name,mimeType,size,modifiedTime,parents)",
        )
        # List only folders directly inside the root folder as series containers.
        root_folders = drive_list(
            "'" + DRIVE_ROOT_FOLDER_ID + "' in parents and trashed = false and mimeType = 'application/vnd.google-apps.folder'",
            "files(id,name,mimeType,modifiedTime,parents)",
        )

        # Keep only actual video files for the standalone movie section.
        movies = []
        for item in root_videos:
            if not is_video_file(item):
                continue
            # Create a stable public key derived from the Drive file ID.
            key = make_key(item["id"])
            # Store only the metadata the website needs.
            movies.append({
                "key": key,
                "name": item["name"],
                "id": item["id"],
                "size": int(item.get("size", 0) or 0),
                "type": "movie",
            })

        # Sort standalone movies naturally by name.
        movies.sort(key=lambda item: item["name"].lower())
        # Prepare the series list.
        series = []

        # Inspect each root-level folder and treat it as one series.
        for folder in root_folders:
            # List only direct children of this series folder.
            episodes_raw = drive_list(
                "'" + folder["id"] + "' in parents and trashed = false and mimeType != 'application/vnd.google-apps.folder'",
                "files(id,name,mimeType,size,modifiedTime,parents)",
            )
            # Keep only supported video files.
            episodes = []
            for item in episodes_raw:
                if not is_video_file(item):
                    continue
                # Generate the same stable key every time this file is discovered.
                key = make_key(item["id"])
                # Store the episode's public metadata.
                episodes.append({
                    "key": key,
                    "name": item["name"],
                    "id": item["id"],
                    "size": int(item.get("size", 0) or 0),
                    "type": "episode",
                    "series_name": folder["name"],
                    "series_id": folder["id"],
                })
            # Ignore empty folders so the homepage stays clean.
            if not episodes:
                continue
            # Sort episodes naturally by filename.
            episodes.sort(key=lambda item: item["name"].lower())
            # Use a stable key for the series itself.
            series_key = make_key("series:" + folder["id"])
            # Add the complete series object.
            series.append({
                "key": series_key,
                "name": folder["name"],
                "id": folder["id"],
                "type": "series",
                "episodes": episodes,
            })

        # Sort series naturally by folder name.
        series.sort(key=lambda item: item["name"].lower())
        # Flatten all video entries for fast key resolution during streaming and progress calls.
        by_key = {}
        for movie in movies:
            by_key[movie["key"]] = movie
        for show in series:
            for episode in show["episodes"]:
                by_key[episode["key"]] = episode

        # Build the final in-memory library object.
        library_cache = {"movies": movies, "series": series, "by_key": by_key}
        # Record when this library was refreshed.
        library_cache_time = time.time()
        # Return the fresh library.
        return library_cache


def get_video(video_key):
    # Normalize the URL key so accidental surrounding spaces do not matter.
    key = video_key.strip().lower()
    # Resolve the key against the current Drive library.
    video = build_library()["by_key"].get(key)
    # Return 404 when the video no longer exists in the Drive library.
    if video is None:
        # Force one refresh because a newly uploaded file may not be in the 60-second cache yet.
        video = build_library(force=True)["by_key"].get(key)
    # Reject unknown or deleted files.
    if video is None:
        raise HTTPException(status_code=404, detail="Video not found")
    # Return both the stable key and its Drive metadata.
    return key, video


def get_metadata(file_id):
    # Read the current time for metadata cache expiry calculations.
    now = time.time()
    # Check the in-memory metadata cache first.
    with metadata_lock:
        cached = metadata_cache.get(file_id)
        if cached and now - cached["time"] < METADATA_TTL:
            return cached["data"]

    # Request only the metadata required by the streaming layer.
    response = get_session().get(
        "https://www.googleapis.com/drive/v3/files/" + file_id,
        params={"fields": "id,name,size,mimeType"},
        headers=drive_headers(),
        timeout=(10, 30),
    )
    # Convert Drive errors into a clean server response.
    if response.status_code != 200:
        response.close()
        raise HTTPException(status_code=502, detail="Google Drive metadata error")
    # Decode the metadata JSON.
    data = response.json()
    # Close the response immediately after decoding it.
    response.close()
    # Streaming requires a known file size.
    if "size" not in data:
        raise HTTPException(status_code=502, detail="Google Drive did not return file size")
    # Normalize the size to an integer.
    data["size"] = int(data["size"])
    # Save the metadata for subsequent requests.
    with metadata_lock:
        metadata_cache[file_id] = {"time": now, "data": data}
    # Return the metadata.
    return data


def get_chunk_lock(file_id, index):
    # Use the Drive ID and chunk number as a unique cache-lock key.
    key = (file_id, index)
    # Protect creation of new lock objects.
    with chunk_locks_guard:
        if key not in chunk_locks:
            # Create a lock only when this chunk is first requested.
            chunk_locks[key] = threading.Lock()
        # Return the lock used by this chunk.
        return chunk_locks[key]


def cache_dir(file_id):
    # Keep each video's cache pieces in its own directory.
    path = CACHE_DIR / file_id
    # Create the directory when the first chunk is needed.
    path.mkdir(parents=True, exist_ok=True)
    # Return the per-video cache directory.
    return path


def chunk_path(file_id, index):
    # Use fixed-width chunk names to keep directory listings predictable.
    return cache_dir(file_id) / ("chunk_%08d.bin" % index)


def evict_cache():
    # Collect all cached chunks and their access times.
    files = []
    # Track total cache bytes.
    total = 0
    # Walk every video cache directory.
    for path in CACHE_DIR.rglob("chunk_*.bin"):
        try:
            # Read file metadata once.
            stat = path.stat()
            # Add this chunk to the total size.
            total += stat.st_size
            # Save access time, path and size for eviction sorting.
            files.append((stat.st_atime, path, stat.st_size))
        except OSError:
            # Ignore chunks that disappear during the scan.
            pass
    # Stop immediately when the cache is below the configured limit.
    if total <= MAX_CACHE_TOTAL:
        return
    # Remove least-recently-used chunks first.
    files.sort(key=lambda item: item[0])
    # Delete old chunks until the cache is back below its limit.
    for _, path, size in files:
        if total <= MAX_CACHE_TOTAL:
            break
        try:
            # Remove the selected cached chunk.
            path.unlink()
            # Keep the running total accurate.
            total -= size
        except OSError:
            # Ignore chunks already removed by another operation.
            pass


def clear_cache():
    # Ensure the cache root exists before deleting its contents.
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    # Iterate over every cached video directory or file.
    for child in CACHE_DIR.iterdir():
        try:
            # Recursively delete video cache directories.
            if child.is_dir():
                shutil.rmtree(child)
            else:
                # Delete unexpected files without touching progress.db.
                child.unlink()
        except OSError:
            # Ignore a file that disappears during cleanup.
            pass


def download_chunk(file_id, index, file_size):
    # Calculate the first byte represented by this cache chunk.
    start = index * CHUNK_SIZE
    # Refuse requests beyond the end of the file.
    if start >= file_size:
        return None
    # Calculate the final byte represented by this chunk.
    end = min(start + CHUNK_SIZE - 1, file_size - 1)
    # Calculate how many bytes the completed chunk must contain.
    expected = end - start + 1
    # Resolve the chunk's persistent cache path.
    path = chunk_path(file_id, index)

    # Prevent multiple viewers from downloading the same chunk simultaneously.
    with get_chunk_lock(file_id, index):
        try:
            # Reuse a complete cached chunk without contacting Drive again.
            if path.exists() and path.stat().st_size == expected:
                # Update access time so active chunks are less likely to be evicted.
                os.utime(path, None)
                # Return the valid cache file.
                return path
        except OSError:
            # Treat a disappearing or inaccessible cache file as a cache miss.
            pass

        # Build the Drive media endpoint for this file.
        url = "https://www.googleapis.com/drive/v3/files/" + file_id
        # Request exactly the byte range needed for this chunk.
        response = get_session().get(
            url,
            params={"alt": "media"},
            headers=drive_headers({"Range": "bytes=%d-%d" % (start, end)}),
            stream=True,
            timeout=(10, 120),
        )
        # Google Drive must return HTTP 206 for a valid byte-range request.
        if response.status_code != 206:
            response.close()
            raise HTTPException(status_code=502, detail="Google Drive range request failed")

        # Validate Drive's Content-Range before writing anything to the cache.
        content_range = response.headers.get("Content-Range", "")
        match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+|\*)", content_range)
        if not match or int(match.group(1)) != start or int(match.group(2)) != end:
            response.close()
            raise HTTPException(status_code=502, detail="Unexpected Google Drive byte range")

        # Write to a temporary file so an interrupted download cannot corrupt a valid chunk.
        temp = path.with_suffix(".tmp")
        written = 0
        try:
            # Stream Drive data to disk in 1 MiB blocks.
            with open(temp, "wb") as output:
                for block in response.iter_content(chunk_size=1024 * 1024):
                    if block:
                        output.write(block)
                        written += len(block)
            # Reject incomplete data before publishing the cache chunk.
            if written != expected:
                raise HTTPException(status_code=502, detail="Incomplete chunk from Google Drive")
            # Atomically publish the complete chunk.
            os.replace(temp, path)
            # Enforce the global cache size limit after the successful download.
            evict_cache()
            # Return the completed cache path.
            return path
        finally:
            # Always close the Google response.
            response.close()
            try:
                # Remove an unfinished temporary file if one remains.
                temp.unlink(missing_ok=True)
            except OSError:
                # Ignore cleanup races.
                pass


def prefetch(file_id, first_index, file_size):
    # Calculate the final valid chunk index for this video.
    last_index = (file_size - 1) // CHUNK_SIZE
    # Download the next few chunks in the background.
    for offset in range(1, PREFETCH_CHUNKS + 1):
        # Calculate the next chunk number.
        index = first_index + offset
        # Stop when the end of the video is reached.
        if index > last_index:
            break
        try:
            # Populate the cache in the background.
            download_chunk(file_id, index, file_size)
        except Exception:
            # Prefetch errors must never interrupt the active playback request.
            pass


def iter_range(file_id, start, end, file_size):
    # Identify the first cache chunk touched by the requested byte range.
    first = start // CHUNK_SIZE
    # Identify the last cache chunk touched by the requested byte range.
    last = end // CHUNK_SIZE
    # Stream each required cache chunk in order.
    for index in range(first, last + 1):
        # Download the chunk on demand or reuse the existing cache file.
        path = download_chunk(file_id, index, file_size)
        # Stop safely if no chunk exists beyond the file size.
        if path is None:
            return
        # Calculate the beginning of this cache chunk in the original video.
        chunk_start = index * CHUNK_SIZE
        # Calculate where the requested range starts inside this chunk.
        offset = max(start, chunk_start) - chunk_start
        # Calculate where the requested range ends inside this chunk.
        limit = min(end, chunk_start + CHUNK_SIZE - 1) - chunk_start
        # Open the cached chunk for streaming.
        with open(path, "rb") as source:
            # Seek directly to the requested byte inside the chunk.
            source.seek(offset)
            # Calculate how many bytes remain to be streamed from this chunk.
            remaining = limit - offset + 1
            # Stream in 1 MiB blocks.
            while remaining > 0:
                # Read no more than 1 MiB at a time.
                block = source.read(min(1024 * 1024, remaining))
                # Stop if the cache file unexpectedly ends.
                if not block:
                    return
                # Reduce the remaining byte count.
                remaining -= len(block)
                try:
                    # Refresh the access time for LRU eviction.
                    os.utime(path, None)
                except OSError:
                    # Ignore access-time failures.
                    pass
                # Yield the block to FastAPI's streaming response.
                yield block


def parse_range(header, file_size):
    # Without a Range header, serve the complete file.
    if not header:
        return 0, file_size - 1
    # Accept the standard single-range syntax used by HTML5 video players.
    match = re.fullmatch(r"bytes=(\d*)-(\d*)", header.strip())
    # Reject malformed or multi-range requests.
    if not match:
        raise HTTPException(status_code=416, detail="Invalid Range header")
    # Extract the optional start and end values.
    start_text, end_text = match.groups()
    # Reject a range that specifies neither boundary.
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
    # Return the normalized byte range.
    return start, end


def init_db():
    # Open the persistent SQLite progress database.
    connection = sqlite3.connect(DB_FILE, timeout=30)
    try:
        # Create the progress table if this is the first startup.
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
        # Persist the schema creation.
        connection.commit()
    finally:
        # Close the SQLite connection.
        connection.close()


# Initialize SQLite before accepting HTTP requests.
init_db()


@app.get("/health")
def health():
    # Read the current library to report its movie and series counts.
    library = build_library()
    # Return simple health information for Railway and manual checks.
    return {
        "status": "ok",
        "movies": len(library["movies"]),
        "series": len(library["series"]),
        "episodes": sum(len(show["episodes"]) for show in library["series"]),
        "chunk_size_mb": CHUNK_SIZE // 1024 // 1024,
        "cache_limit_gb": round(MAX_CACHE_TOTAL / 1024 / 1024 / 1024, 2),
    }


@app.get("/api/videos")
def api_videos():
    # Read the dynamic Google Drive library.
    library = build_library()
    # Return public metadata without exposing Drive IDs.
    return {
        "movies": [
            {"key": item["key"], "name": item["name"], "size": item["size"]}
            for item in library["movies"]
        ],
        "series": [
            {
                "key": show["key"],
                "name": show["name"],
                "episodes": [
                    {"key": ep["key"], "name": ep["name"], "size": ep["size"]}
                    for ep in show["episodes"]
                ],
            }
            for show in library["series"]
        ],
    }


@app.post("/api/refresh")
def refresh_library():
    # Force the server to rescan the Google Drive root folder immediately.
    library = build_library(force=True)
    # Report the newly discovered counts.
    return {
        "ok": True,
        "movies": len(library["movies"]),
        "series": len(library["series"]),
        "episodes": sum(len(show["episodes"]) for show in library["series"]),
    }


@app.post("/admin/clear-cache")
def admin_clear_cache(request: Request):
    # Refuse the endpoint entirely unless a secret token has been configured.
    if not CACHE_ADMIN_TOKEN:
        raise HTTPException(status_code=404, detail="Not found")
    # Read the token from a custom HTTP header.
    supplied = request.headers.get("X-Cache-Admin-Token", "")
    # Reject missing or incorrect tokens.
    if supplied != CACHE_ADMIN_TOKEN:
        raise HTTPException(status_code=403, detail="Forbidden")
    # Delete only video cache data and keep progress.db intact.
    clear_cache()
    # Confirm successful cleanup.
    return {"ok": True, "message": "Video cache cleared"}


@app.get("/api/progress/{video_key}")
def get_progress(video_key: str, viewer_id: str = ""):
    # Validate that the video key currently exists.
    key, _ = get_video(video_key)
    # Limit the browser-provided viewer ID to a safe length.
    viewer_id = viewer_id.strip()[:128]
    # Return zero progress when the browser has no viewer identifier.
    if not viewer_id:
        return {"position": 0, "duration": 0}
    # Open the persistent SQLite database.
    connection = sqlite3.connect(DB_FILE, timeout=30)
    try:
        # Read the last saved playback position for this viewer and video.
        row = connection.execute(
            "SELECT position, duration FROM progress WHERE viewer_id=? AND video_key=?",
            (viewer_id, key),
        ).fetchone()
    finally:
        # Always close the database connection.
        connection.close()
    # Return empty progress when this viewer has never watched the video.
    if row is None:
        return {"position": 0, "duration": 0}
    # Return the stored playback position and duration.
    return {"position": row[0], "duration": row[1]}


@app.post("/api/progress/{video_key}")
async def save_progress(video_key: str, request: Request):
    # Validate that the video key currently exists.
    key, _ = get_video(video_key)
    try:
        # Parse the JSON body sent by the browser.
        data = await request.json()
    except Exception:
        # Reject malformed progress requests.
        raise HTTPException(status_code=400, detail="Invalid JSON")
    # Read and limit the browser-specific viewer identifier.
    viewer_id = str(data.get("viewer_id", "")).strip()[:128]
    # A viewer identifier is required for progress storage.
    if not viewer_id:
        raise HTTPException(status_code=400, detail="viewer_id is required")
    try:
        # Normalize playback position and duration to non-negative numbers.
        position = max(0.0, float(data.get("position", 0)))
        duration = max(0.0, float(data.get("duration", 0)))
    except (TypeError, ValueError):
        # Reject invalid numeric values.
        raise HTTPException(status_code=400, detail="Invalid position or duration")
    # Treat a video within two seconds of its end as completed.
    if duration > 0 and position >= max(duration - 2, 0):
        position = 0.0
    # Open the persistent SQLite database.
    connection = sqlite3.connect(DB_FILE, timeout=30)
    try:
        # Insert or update this viewer's progress for this video.
        connection.execute(
            "INSERT INTO progress (viewer_id, video_key, position, duration, updated_at) "
            "VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(viewer_id, video_key) DO UPDATE SET "
            "position=excluded.position, duration=excluded.duration, updated_at=excluded.updated_at",
            (viewer_id, key, position, duration, int(time.time())),
        )
        # Persist the progress update.
        connection.commit()
    finally:
        # Close the database connection.
        connection.close()
    # Confirm success to the browser.
    return {"ok": True}


@app.get("/video/{video_key}")
def stream_video(video_key: str, request: Request):
    # Resolve the stable public key to the current Drive file.
    _, video = get_video(video_key)
    # Read current Drive metadata, using the metadata cache where possible.
    metadata = get_metadata(video["id"])
    # Read the exact byte size required for Range handling.
    file_size = metadata["size"]
    # Convert the browser Range header into concrete byte offsets.
    start, end = parse_range(request.headers.get("range"), file_size)
    # Return HTTP 206 when the browser explicitly requested a range.
    status = 206 if request.headers.get("range") else 200
    # Prefetch upcoming chunks in the background.
    executor.submit(prefetch, video["id"], start // CHUNK_SIZE, file_size)
    # Tell the browser that this endpoint supports byte ranges.
    headers = {
        "Accept-Ranges": "bytes",
        "Content-Length": str(end - start + 1),
        "Content-Type": metadata.get("mimeType", "video/mp4"),
        "Cache-Control": "public, max-age=3600",
        "X-Accel-Buffering": "no",
    }
    # Add the mandatory Content-Range header for HTTP 206.
    if status == 206:
        headers["Content-Range"] = "bytes %d-%d/%d" % (start, end, file_size)
    # Stream only the requested byte range to the browser.
    return StreamingResponse(
        iter_range(video["id"], start, end, file_size),
        status_code=status,
        headers=headers,
        media_type=metadata.get("mimeType", "video/mp4"),
    )


@app.head("/video/{video_key}")
def head_video(video_key: str):
    # Resolve the requested public video key.
    _, video = get_video(video_key)
    # Read current Drive metadata.
    metadata = get_metadata(video["id"])
    # Return only the headers required by a browser HEAD request.
    return Response(
        status_code=200,
        headers={
            "Accept-Ranges": "bytes",
            "Content-Length": str(metadata["size"]),
            "Content-Type": metadata.get("mimeType", "video/mp4"),
        },
    )


# Escape arbitrary Drive names before inserting them into HTML.
def safe_text(value):
    # Convert the value to a string and HTML-escape it.
    return html.escape(str(value), quote=True)


# Create a simple dark player page that keeps the existing resume functionality.
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
.back { display:inline-block; margin-bottom:12px; color:#aaa; text-decoration:none; }
</style>
</head>
<body>
<div class="wrap">
<a class="back" href="/">← Thư viện</a>
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
    const payload = JSON.stringify({viewer_id: viewerId, position: position, duration: duration});
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
player.addEventListener("timeupdate", function() { saveProgress(false); });
player.addEventListener("pause", function() { saveProgress(true); });
player.addEventListener("ended", function() {
    localStorage.removeItem(positionStorageKey);
    fetch("/api/progress/" + VIDEO_KEY, {
        method:"POST",
        headers:{"Content-Type":"application/json"},
        body:JSON.stringify({viewer_id:viewerId, position:0, duration:player.duration}),
        keepalive:true
    }).catch(function() {});
});
window.addEventListener("beforeunload", function() {
    if (!Number.isFinite(player.currentTime)) return;
    const payload = JSON.stringify({viewer_id:viewerId, position:player.currentTime || 0, duration:player.duration || 0});
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
    # Resolve the requested video from the dynamic Drive library.
    key, video = get_video(video_key)
    # Build the navigation list for episodes in the same series.
    buttons = ""
    if video.get("type") == "episode":
        # Read the current library so the series episode list stays dynamic.
        library = build_library()
        # Find the matching series by its stable Drive-derived ID.
        current_series = next(
            (show for show in library["series"] if show["id"] == video.get("series_id")),
            None,
        )
        if current_series:
            # Create a link for every episode in the current series.
            buttons = "".join(
                '<a class="item%s" href="/watch/%s">%s</a>'
                % (
                    " active" if episode["key"] == key else "",
                    episode["key"],
                    safe_text(episode["name"]),
                )
                for episode in current_series["episodes"]
            )
    # Standalone movies get no episode bar.
    if not buttons:
        buttons = '<a class="item active" href="/watch/%s">%s</a>' % (
            key,
            safe_text(video["name"]),
        )
    # Replace placeholders without using Python f-strings so JavaScript braces remain untouched.
    page = PAGE_TEMPLATE.replace("__TITLE__", safe_text(video["name"]))
    page = page.replace("__VIDEO_KEY__", key)
    page = page.replace("__BUTTONS__", buttons)
    # Return the finished player page.
    return HTMLResponse(page)


@app.get("/", response_class=HTMLResponse)
def index():
    # Read the current Google Drive library.
    library = build_library()
    # Build standalone movie cards.
    movie_items = "".join(
        '<li><a href="/watch/%s">%s</a></li>' % (item["key"], safe_text(item["name"]))
        for item in library["movies"]
    )
    # Build series sections with their episode lists.
    series_items = "".join(
        '<li><strong>%s</strong><ul>%s</ul></li>'
        % (
            safe_text(show["name"]),
            "".join(
                '<li><a href="/watch/%s">%s</a></li>'
                % (episode["key"], safe_text(episode["name"]))
                for episode in show["episodes"]
            ),
        )
        for show in library["series"]
    )
    # Return a simple homepage that clearly separates standalone movies and series.
    return HTMLResponse(
        "<!doctype html><html lang='vi'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>Videos</title>"
        "<style>body{font-family:Arial,sans-serif;background:#111;color:#eee;margin:0;padding:20px;}"
        ".wrap{max-width:900px;margin:auto;}a{color:#eee;}li{margin:7px 0;}"
        "button{padding:8px 12px;margin-bottom:15px;cursor:pointer;}</style></head><body>"
        "<div class='wrap'><h1>Thư viện video</h1>"
        "<button onclick='refreshLibrary()'>↻ Làm mới thư viện</button>"
        "<h2>Phim lẻ</h2><ul>" + (movie_items or "<li>Chưa có phim lẻ.</li>") + "</ul>"
        "<h2>Phim bộ</h2><ul>" + (series_items or "<li>Chưa có phim bộ.</li>") + "</ul>"
        "<p><a href='/health'>Server health</a></p></div>"
        "<script>async function refreshLibrary(){"
        "const button=document.querySelector('button');button.disabled=true;button.textContent='Đang làm mới...';"
        "try{await fetch('/api/refresh',{method:'POST'});location.reload();}"
        "catch(e){button.disabled=false;button.textContent='Lỗi - thử lại';}}"
        "</script></body></html>"
    )
