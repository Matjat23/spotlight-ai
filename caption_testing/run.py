"""From the project root: python -m caption_testing.run --help"""
import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from caption_testing.common import DEFAULT_PROMPT, FIELDS, discover_photos
from caption_testing.registry import MODELS, adapter_class

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def positive_int(text):
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("Must be at least 1")
    return value

def package_version(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return "not installed"

def make_parser():
    p = argparse.ArgumentParser(description="Local caption comparison: one process per model, one CSV per run.")
    p.add_argument("--folder", type=Path)
    p.add_argument("--models", nargs="+", choices=list(MODELS), default=["blip", "florence-base"])
    p.add_argument("--list-models", action="store_true")
    p.add_argument("--limit", type=positive_int, help="Photos per model; default all")
    p.add_argument("--recursive", action="store_true", help="Include subfolders")
    p.add_argument("--threads", type=positive_int, default=2)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--max-new-tokens", type=positive_int, help="Override each model's default")
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--florence-task", choices=["<CAPTION>", "<DETAILED_CAPTION>", "<MORE_DETAILED_CAPTION>"], default="<MORE_DETAILED_CAPTION>")
    p.add_argument("--qwen-min-pixels", type=positive_int, default=256*28*28)
    p.add_argument("--qwen-max-pixels", type=positive_int, default=512*28*28)
    p.add_argument("--offline", action="store_true")
    p.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "caption_tests")
    p.add_argument("--_worker", nargs=2, help=argparse.SUPPRESS)
    return p

def run_worker(manifest_path, key):
    config = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    os.environ["OMP_NUM_THREADS"] = str(config["threads"])
    os.environ["MKL_NUM_THREADS"] = str(config["threads"])
    if config["offline"]:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    from caption_testing.common import load_image, synchronize
    torch.set_num_threads(config["threads"])
    torch.manual_seed(0)
    device = config["device"]
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable. Use --device cpu or install CUDA PyTorch.")
    spec = MODELS[key]
    dtype = "float16" if device == "cuda" and spec.module not in {"blip", "florence"} else "float32"
    settings = dict(config, device=device, dtype=dtype,
                    max_new_tokens=config["max_new_tokens"] or spec.default_tokens, num_beams=spec.beams)
    try:
        captioner = adapter_class(spec)(spec, settings)
    except Exception as error:
        with Path(config["csv_path"]).open("a", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=FIELDS).writerow(dict(
                batch=Path(config["folder"]).name, model=spec.model_id, model_key=key,
                run_id=config["run_id"], status="load_error",
                error=f"{type(error).__name__}: {error}"))
        print(f"ADAPTER IMPORT ERROR: {error}", flush=True)
        return 1
    folder = Path(config["folder"])
    base = dict(batch=folder.name, model=spec.model_id, task=captioner.task,
                model_key=key, device=device, dtype=dtype, threads=config["threads"],
                max_new_tokens=settings["max_new_tokens"], num_beams=spec.beams,
                revision=spec.revision, run_id=config["run_id"],
                prompt=config["prompt"] if spec.module not in {"blip", "florence"} else "",
                qwen_min_pixels=config["qwen_min_pixels"] if key == "qwen-vl" else "",
                qwen_max_pixels=config["qwen_max_pixels"] if key == "qwen-vl" else "",
                torch_version=torch.__version__, transformers_version=package_version("transformers"))
    print(f"\nLoading {spec.model_id} on {device} ({dtype})...", flush=True)
    if key == "moondream2":
        print("Using the official Moondream custom code, revision 2025-06-21.", flush=True)
    started = time.perf_counter()
    with Path(config["csv_path"]).open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        try:
            captioner.load()
            synchronize(device)
        except Exception as error:
            writer.writerow(dict(base, status="load_error", error=f"{type(error).__name__}: {error}"))
            handle.flush()
            print(f"MODEL LOAD ERROR: {error}", flush=True)
            return 1
        base["model_load_seconds"] = round(time.perf_counter() - started, 3)
        base["resolved_revision"] = captioner.resolved_revision()
        failed = False
        for index, name in enumerate(config["photos"], 1):
            print(f"[{index}/{len(config['photos'])}] {name} - generating...", flush=True)
            row = dict(base, photo=name)
            image = None
            try:
                image = load_image(folder / name)
                row.update(width=image.width, height=image.height)
                synchronize(device)
                started = time.perf_counter()
                with torch.inference_mode():
                    caption = captioner.caption(image)
                synchronize(device)
                row.update(caption=caption, seconds=round(time.perf_counter() - started, 3), status="ok")
                print(caption, flush=True)
                print(f"Processing time: {row['seconds']:.3f} seconds\n", flush=True)
            except Exception as error:
                failed = True
                row.update(status="image_error", error=f"{type(error).__name__}: {error}")
                print(f"IMAGE ERROR: {error}", flush=True)
            finally:
                if image is not None:
                    image.close()
            writer.writerow(row)
            handle.flush()
    return int(failed)

def main(argv=None):
    parser = make_parser()
    args = parser.parse_args(argv)
    if args._worker:
        return run_worker(*args._worker)
    if args.list_models:
        for key, spec in MODELS.items():
            print(f"{key:20} {spec.model_id}")
        return 0
    if args.folder is None:
        parser.error("--folder is required unless using --list-models")
    if args.qwen_min_pixels > args.qwen_max_pixels:
        parser.error("--qwen-min-pixels must not exceed --qwen-max-pixels")
    try:
        photos = discover_photos(args.folder, args.recursive, args.limit)
    except ValueError as error:
        parser.error(str(error))
    folder = args.folder.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    csv_path = output_dir / f"caption_test_{run_id}.csv"
    manifest = output_dir / f"caption_test_{run_id}.json"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        csv.DictWriter(handle, fieldnames=FIELDS).writeheader()
    config = vars(args).copy()
    config.pop("_worker")
    config.update(folder=str(folder), output_dir=str(output_dir), run_id=run_id,
                  csv_path=str(csv_path), photos=[p.relative_to(folder).as_posix() for p in photos],
                  models=list(dict.fromkeys(args.models)),
                  package_versions={p: package_version(p) for p in ["torch", "transformers", "pillow-heif", "accelerate"]})
    manifest.write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"Found {len(photos)} photos. CPU threads: {args.threads}.", flush=True)
    print(f"Results: {csv_path}\nOne process per model. Ctrl+C stops the run.", flush=True)
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    failed = False
    for key in config["models"]:
        process = subprocess.Popen(
            [sys.executable, "-m", "caption_testing.run", "--_worker", str(manifest), key],
            cwd=PROJECT_ROOT, env=environment)
        try:
            code = process.wait()
        except KeyboardInterrupt:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            print(f"\nStopped. Completed rows remain in {csv_path}", flush=True)
            return 130
        if code:
            failed = True
            print(f"{key} reported an error (exit {code}). Check CSV/terminal; continuing.", flush=True)
    print(f"\nResults saved to: {csv_path}\nSettings saved to: {manifest}", flush=True)
    return int(failed)

if __name__ == "__main__":
    raise SystemExit(main())
