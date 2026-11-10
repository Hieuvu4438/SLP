"""Fetch/verify author assets. Never use an existing experiment run as a checkpoint source.
A local download cache is reused only when its SHA256 equals the author's Hub LFS object ID.
"""

import argparse
import hashlib
import json
from pathlib import Path
import zipfile
import requests

REPO = "2000zrl/MaDiS"
FILES = ("tokenizer.ckpt", "sft_csl.ckpt", "mean.pt", "std.pt", "Qwen3-0.6B-Base-en-zh-de.zip")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for data in iter(lambda: f.read(1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--upstream-repo", required=True, help="Fresh official MaDiS source checkout"
    )
    parser.add_argument("--local-download-cache")
    args = parser.parse_args(argv)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    response = requests.get(
        f"https://huggingface.co/api/models/{REPO}/tree/main",
        params={"recursive": "true"},
        timeout=30,
    )
    response.raise_for_status()
    metadata = {entry["path"]: entry for entry in response.json()}
    log = []
    for name in FILES:
        expected = metadata[name]["lfs"]["oid"]
        path = output / name
        cached = Path(args.local_download_cache) / name if args.local_download_cache else None
        if path.exists() and digest(path) == expected:
            source = "already_verified"
        elif cached and cached.exists() and digest(cached) == expected:
            if path.exists() or path.is_symlink():
                path.unlink()
            path.symlink_to(cached.resolve())
            source = "verified_author_download_cache"
        else:
            from huggingface_hub import hf_hub_download

            downloaded = Path(hf_hub_download(REPO, name))
            if digest(downloaded) != expected:
                raise ValueError("Author LFS SHA256 mismatch")
            if path.exists() or path.is_symlink():
                path.unlink()
            path.symlink_to(downloaded.resolve())
            source = "fresh_author_Hub_download"
        log.append(dict(file=name, sha256=expected, source=source, repository=REPO))
        print(f"{name}: verified ({source})", flush=True)
    archive = output / FILES[-1]
    model_root = output / "qwen"
    model_root.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        for item in z.infolist():
            destination = (model_root / item.filename).resolve()
            if not destination.is_relative_to(model_root.resolve()):
                raise ValueError("Invalid archive path")
        z.extractall(model_root)
    folders = [p.parent for p in model_root.rglob("map_ids.pkl")]
    if len(folders) != 1:
        raise ValueError("Cannot identify the author Qwen model folder")
    config = dict(
        base_type="madis",
        target_language="csl",
        upstream_repo=str(Path(args.upstream_repo).resolve()),
        tokenizer_checkpoint=str(output / "tokenizer.ckpt"),
        generator_checkpoint=str(output / "sft_csl.ckpt"),
        model_path=str(folders[0]),
        soke_mean=str(output / "mean.pt"),
        soke_std=str(output / "std.pt"),
        device="cuda",
    )
    (output / "assets.json").write_text(json.dumps(config, indent=2) + "\n")
    (output / "download_provenance.json").write_text(json.dumps(log, indent=2) + "\n")
    print(output / "assets.json")


if __name__ == "__main__":
    main()
