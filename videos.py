"""Gets videos ready for search. Run this before pipeline.py.

For every video it:
1. makes a small H.264 copy that every browser can play (iPhone .MOV is often HEVC, which Chrome on
   Windows cannot play), in data/videos/web
2. takes one frame every few seconds and keeps only the frames where the scene changed, saved in
   data/raw as zz_video_V01_00012.jpg (video V01, second 12)
3. writes down what is said with Whisper (runs on this laptop, the audio never leaves it), with the
   time of every sentence, saved in outputs/videos.json

pipeline.py then describes those frames like any photo, so search finds the moment inside a video.
The "zz_" makes the frames sort after the photos, so photo numbers do not change.

    python videos.py                          (videos already in data/videos)
    python videos.py C:/path/to/Videos.zip    (copies the videos out of the zip first)
    python videos.py --speech-again           (writes the speech down again, for example with a new model)
"""
import json
import shutil
import subprocess
import sys
import tempfile
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).parent
VIDEOS = ROOT / "data" / "videos"
WEB = VIDEOS / "web"
RAW = ROOT / "data" / "raw"
INFO = ROOT / "outputs" / "videos.json"
EXTS = {".mov", ".mp4", ".m4v", ".avi"}
EVERY = 2          # seconds between frames we look at
MIN_CHANGE = 12    # mean pixel change (0-255) on a tiny gray copy before a frame counts as a new scene
MAX_GAP = 10       # keep a frame at least every 10 seconds, even when little changes
MAX_FRAMES = 40    # per video, so one long video does not take over the archive
WHISPER = "openai/whisper-small"  # 244M parameters, about 1 GB of GPU memory, many languages
# Whisper sometimes "hears" these in silence or music, they are not real speech
NOT_SPEECH = {"", "you", "thank you", "thank you.", "thanks for watching!", "thanks for watching.", "bye.", "..."}


def run(*cmd):
    return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout


def probe(path):
    """Length in seconds and when it was filmed (or None)."""
    info = json.loads(run("ffprobe", "-v", "error", "-print_format", "json", "-show_format", str(path)))
    fmt = info["format"]
    tags = {k.lower(): v for k, v in fmt.get("tags", {}).items()}
    when = tags.get("com.apple.quicktime.creationdate") or tags.get("creation_time")
    try:
        when = datetime.fromisoformat(when.replace("Z", "+00:00")).replace(tzinfo=None) if when else None
    except ValueError:
        when = None
    return float(fmt["duration"]), when


def unzip(zip_path):
    VIDEOS.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        for item in z.infolist():
            name = Path(item.filename).name
            if Path(name).suffix.lower() in EXTS and not (VIDEOS / name).exists():
                with z.open(item) as src, open(VIDEOS / name, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                print(f"  copied {name}")


def web_copy(src, dst):
    """720p H.264 with the index at the start, so the browser can start playing and jump right away.
    ffmpeg turns the picture upright from the phone's rotation tag."""
    run("ffmpeg", "-y", "-v", "error", "-i", str(src), "-vf", "scale=-2:'min(720,ih)'",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "26", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "96k", "-movflags", "+faststart", str(dst))


def key_frames(src, vid, when):
    """Saves the frames where something changed. Returns how many were kept."""
    for old in RAW.glob(f"zz_video_{vid}_*.jpg"):
        old.unlink()
    with tempfile.TemporaryDirectory() as tmp:
        run("ffmpeg", "-v", "error", "-i", str(src), "-vf", f"fps=1/{EVERY},scale='min(1280,iw)':-2",
            "-q:v", "3", str(Path(tmp) / "%05d.jpg"))
        frames = sorted(Path(tmp).glob("*.jpg"))
        kept, last, last_t = [], None, -MAX_GAP
        for n, f in enumerate(frames):
            t = n * EVERY
            img = Image.open(f)
            small = np.asarray(img.convert("L").resize((32, 32)), dtype=np.float32)
            if last is None or np.abs(small - last).mean() >= MIN_CHANGE or t - last_t >= MAX_GAP:
                kept.append((t, img))
                last, last_t = small, t
        if len(kept) > MAX_FRAMES:  # spread the frames we keep over the whole video
            kept = [kept[round(i)] for i in np.linspace(0, len(kept) - 1, MAX_FRAMES)]
        for t, img in kept:
            exif = Image.Exif()
            if when:  # the date goes in EXIF, so the Year filter and event grouping work for videos too
                exif.get_ifd(0x8769)[36867] = (when + timedelta(seconds=t)).strftime("%Y:%m:%d %H:%M:%S")
            img.convert("RGB").save(RAW / f"zz_video_{vid}_{t:05d}.jpg", quality=90, exif=exif)
    return len(kept)


def audio(src):
    """The sound track as 16 kHz mono numbers, what Whisper expects. None when the video has no sound."""
    out = subprocess.run(["ffmpeg", "-v", "error", "-i", str(src), "-vn", "-ac", "1", "-ar", "16000",
                          "-f", "f32le", "-"], capture_output=True).stdout
    return np.frombuffer(out, dtype=np.float32) if out else None


_whisper = None


def speech(src):
    """What is said, as [start second, end second, text]."""
    global _whisper
    sound = audio(src)
    if sound is None or len(sound) < 16000:
        return []
    if _whisper is None:
        import torch
        from transformers import pipeline as hf_pipeline
        cuda = torch.cuda.is_available()
        _whisper = hf_pipeline("automatic-speech-recognition", model=WHISPER, device=0 if cuda else -1,
                               torch_dtype=torch.float16 if cuda else torch.float32)
    # Whisper's own long-form mode with its guards against loops: retry with more randomness when a
    # piece repeats itself (compression ratio) or the model is unsure, and skip pieces that are silence
    result = _whisper({"raw": sound, "sampling_rate": 16000}, return_timestamps=True,
                      generate_kwargs={"task": "transcribe", "condition_on_prev_tokens": False,
                                       "compression_ratio_threshold": 1.35, "logprob_threshold": -1.0,
                                       "no_speech_threshold": 0.6, "temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)})
    lines, seen = [], set()
    for c in result.get("chunks", []):
        start, end = c["timestamp"]
        start = start or 0
        end = end if end is not None else start + 2
        text = c["text"].strip()
        too_fast = len(text.split()) > 6 * max(end - start, 1)  # people say about 2-3 words a second
        if text.lower() in NOT_SPEECH or text.lower() in seen or too_fast:
            continue
        seen.add(text.lower())
        lines.append([round(start, 1), round(end, 1), text])
    return lines


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    again = "--speech-again" in sys.argv
    if args:
        unzip(args[0])
    sources = sorted(p for p in VIDEOS.glob("*") if p.suffix.lower() in EXTS)
    if not sources:
        raise SystemExit(f"No videos in {VIDEOS}")
    WEB.mkdir(parents=True, exist_ok=True)
    INFO.parent.mkdir(exist_ok=True)
    info = json.loads(INFO.read_text(encoding="utf-8")) if INFO.exists() else {}
    by_source = {v["source"]: vid for vid, v in info.items()}

    for src in sources:
        # A video keeps its number when more videos are added later
        vid = by_source.get(src.name) or f"V{len(info) + 1:02d}"
        duration, when = probe(src)
        web = WEB / f"{vid}.mp4"
        if not web.exists():
            print(f"{vid}: making a web copy ({duration:.0f} s)", flush=True)
            web_copy(src, web)
        frames = key_frames(src, vid, when)
        said = info.get(vid, {}).get("speech")
        if said is None or again:
            said = speech(src)
        info[vid] = {"source": src.name, "web": web.name, "duration": round(duration, 1),
                     "taken": when.isoformat(sep=" ") if when else "", "frames": frames, "speech": said}
        print(f"{vid}: {frames} moments, {len(said)} spoken sentences", flush=True)

    INFO.write_text(json.dumps(info, indent=1), encoding="utf-8")
    total = sum(v["frames"] for v in info.values())
    print(f"Done: {len(info)} videos, {total} moments. Now run python pipeline.py to describe them.")


if __name__ == "__main__":
    main()
