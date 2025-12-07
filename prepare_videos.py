import os
import sys
import subprocess
import argparse
import platform
import urllib.request
import zipfile
import tarfile
from multiprocessing import Pool, cpu_count
from pathlib import Path

# --- Video Re-encoding Logic ---

TARGET_SCALE   = "256:256"   # WxH
CRF            = "25"        # x264 quality
PRESET         = "superfast" # x264 preset
DISABLE_AUDIO  = True
FOLLOW_SYMLINKS = False

FFMPEG_URLS = {
    "Windows": "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip",
    "Darwin":  "https://evermeet.cx/ffmpeg/ffmpeg-7.1.zip",
    "Linux":   "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz",
}

VIDEO_EXTS = {
    ".mts", ".m2ts", ".mp4", ".mov", ".m4v", ".avi", ".mkv", ".wmv",
    ".mpg", ".mpeg", ".mp2", ".mpv", ".3gp", ".3g2", ".webm",
}

def get_platform():
    sysname = platform.system()
    return sysname if sysname in ("Windows", "Darwin", "Linux") else "Linux"

def download_file(url, filename):
    print(f"Downloading {filename}...")
    def progress_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100, (downloaded * 100) // total_size)
            print(f"\rProgress: {percent}% ({downloaded}/{total_size} bytes)", end="")
    urllib.request.urlretrieve(url, filename, reporthook=progress_hook)
    print("\nDownload complete!")

def extract_ffmpeg(archive_path, extract_dir):
    print(f"Extracting {archive_path}...")
    if archive_path.endswith(".zip"):
        with zipfile.ZipFile(archive_path, "r") as z:
            z.extractall(extract_dir)
    elif archive_path.endswith(".tar.xz"):
        with tarfile.open(archive_path, "r:xz") as t:
            t.extractall(extract_dir)
    else:
        raise RuntimeError(f"Unsupported archive format: {archive_path}")
    print("Extraction complete!")

def find_ffmpeg_binary(extract_dir, platform_name):
    if platform_name == "Windows":
        for root, _, files in os.walk(extract_dir):
            if "ffmpeg.exe" in files:
                return os.path.join(root, "ffmpeg.exe")
    else:
        for root, _, files in os.walk(extract_dir):
            if "ffmpeg" in files:
                ffmpeg_path = os.path.join(root, "ffmpeg")
                try:
                    os.chmod(ffmpeg_path, 0o755)
                except Exception:
                    pass
                return ffmpeg_path
    return None

def setup_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        print("✅ FFmpeg is already installed and available!")
        return "ffmpeg"
    except Exception:
        print("FFmpeg not found in PATH. Setting up FFmpeg automatically...")

    platform_name = get_platform()
    if platform_name not in FFMPEG_URLS:
        raise RuntimeError(f"Unsupported platform: {platform_name}")

    ffmpeg_dir = os.path.join(os.getcwd(), "ffmpeg_portable")
    os.makedirs(ffmpeg_dir, exist_ok=True)

    existing_ffmpeg = find_ffmpeg_binary(ffmpeg_dir, platform_name)
    if existing_ffmpeg and os.path.exists(existing_ffmpeg):
        print("✅ Using existing portable FFmpeg installation!")
        return existing_ffmpeg

    url = FFMPEG_URLS[platform_name]
    archive_name = url.split("/")[-1]
    archive_path = os.path.join(ffmpeg_dir, archive_name)

    try:
        download_file(url, archive_path)
        extract_ffmpeg(archive_path, ffmpeg_dir)
        ffmpeg_binary = find_ffmpeg_binary(ffmpeg_dir, platform_name)
        if not ffmpeg_binary:
            raise RuntimeError("Could not find FFmpeg binary after extraction")

        subprocess.run([ffmpeg_binary, "-version"], capture_output=True, check=True)
        print("✅ FFmpeg setup complete!")
        try:
            os.remove(archive_path)
        except Exception:
            pass
        return ffmpeg_binary
    except Exception as e:
        print(f"❌ Error setting up FFmpeg: {e}")
        sys.exit(1)

def iter_video_files(root):
    root_p = Path(root).resolve()
    for dirpath, dirnames, filenames in os.walk(root_p, followlinks=FOLLOW_SYMLINKS):
        for fname in filenames:
            if Path(fname).suffix.lower() in VIDEO_EXTS:
                abs_p = Path(dirpath) / fname
                rel_p = abs_p.relative_to(root_p).as_posix()
                yield str(abs_p), rel_p

def build_ffmpeg_cmd(ffmpeg_bin, input_path, output_path):
    cmd = [
        ffmpeg_bin, "-i", input_path,
        "-vf", f"scale={TARGET_SCALE}:flags=lanczos",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", CRF,
        "-preset", PRESET,
    ]
    if DISABLE_AUDIO:
        cmd += ["-an"]
    cmd += ["-y", output_path]
    return cmd

def _process_one(args):
    input_abs, rel_posix, output_dir, ffmpeg_bin, preserve_subfolders = args
    in_stem = Path(rel_posix).stem
    out_dir = Path(output_dir)
    if preserve_subfolders:
        # Replicate subfolder structure
        parent = Path(rel_posix).parent
        out_dir = out_dir / parent
    
    out_dir.mkdir(parents=True, exist_ok=True)

    output_path = str((out_dir / f"{in_stem}.mp4").resolve())
    counter = 1
    while os.path.exists(output_path):
        output_path = str((out_dir / f"{in_stem}_{counter}.mp4").resolve())
        counter += 1

    cmd = build_ffmpeg_cmd(ffmpeg_bin, input_abs, output_path)
    try:
        print(f"Processing: {rel_posix} -> {Path(output_path).name}")
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"✅ Done: {Path(output_path).name}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error processing {rel_posix}:\n{e.stderr}")
        return False

def reencode_videos(input_dir, output_dir, processes=4, preserve_subfolders=True):
    print("🎬 FERAL Video Re-encoding")
    print("=" * 50)
    in_dir = Path(input_dir)
    out_dir = Path(output_dir)

    if not in_dir.exists() or not in_dir.is_dir():
        print(f"❌ Input directory does not exist or is not a directory: {in_dir}")
        return

    items = list(iter_video_files(str(in_dir)))
    if not items:
        print("⚠️  No video files found.")
        return

    print(f"📁 Found {len(items)} video files to process")
    out_dir.mkdir(parents=True, exist_ok=True)

    ffmpeg_bin = setup_ffmpeg()
    max_procs = max(1, min(processes, cpu_count()))

    args_list = [(abs_p, rel_p, str(out_dir), ffmpeg_bin, preserve_subfolders) for abs_p, rel_p in items]

    with Pool(processes=max_procs) as pool:
        results = pool.map(_process_one, args_list)

    ok = sum(1 for r in results if r)
    print(f"🎉 Completed: {ok}/{len(results)} succeeded")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FERAL Video Re-encoder")
    parser.add_argument("input_dir", help="Input directory containing videos")
    parser.add_argument("output_dir", help="Output directory for re-encoded videos")
    parser.add_argument("--no-preserve", action="store_true", help="Flatten folder structure")
    
    args = parser.parse_args()
    reencode_videos(args.input_dir, args.output_dir, preserve_subfolders=not args.no_preserve)
