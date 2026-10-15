#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# MIT License
#
# Copyright (c) 2024 fake-news-jev contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
"""Download (or print instructions for) the four fake-news datasets.

Datasets:

* ``fake_real``  — Kaggle "Fake and real news dataset" (Clement Bisaillon).
                   Kaggle credentials (``~/.kaggle/kaggle.json`` or env
                   ``KAGGLE_USERNAME`` / ``KAGGLE_KEY``) required.
* ``liar``       — LIAR (UCSB Politifact). Direct TSV download from
                   https://www.cs.ucsb.edu/~william/data/liar_dataset.zip —
                   fetched automatically when network is available.
* ``fakenewsnet``— FakeNewsNet (KaiDMML). Source of truth is the GitHub
                   repo https://github.com/KaiDMML/FakeNewsNet, distributed
                   as per-article JSON files. We print the clone + flatten
                   instructions; a flattened CSV is also reachable from a
                   community mirror if you set ``FNN_MIRROR_URL``.
* ``welfake``    — WELFake (Kaggle, 72k articles). Kaggle credentials
                   required (same as ``fake_real``).

This script is OFFLINE-SAFE: it never hard-fails if a dataset can't be
fetched. It will:

  1. Print the manual instructions for any dataset it couldn't fetch.
  2. Skip Kaggle datasets gracefully when no Kaggle creds are available.
  3. Skip LIAR's direct download if the network is unreachable.
  4. Exit 0 even if zero datasets were fetched (downstream loaders fall
     back to the synthetic dataset generator for tests).

It is the ONLY file in the repo that may touch the network; the core
loaders in ``src/fakenews/data.py`` only read from local paths.

Usage:
    python scripts/download_data.py                    # all datasets
    python scripts/download_data.py --dataset liar     # one dataset
    python scripts/download_data.py --out ./data/raw   # custom output dir
    python scripts/download_data.py --list             # just list URLs

Exit codes:
    0 — script ran (may have printed manual instructions instead of downloading)
    2 — bad CLI args (argparse default)
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional

DEFAULT_OUT = Path(__file__).resolve().parents[1] / "data" / "raw"

# Per-dataset metadata for the downloader.
DATASETS: dict[str, dict[str, str]] = {
    "fake_real": {
        "kaggle_slug": "clmentbisaillon/fake-and-real-news-dataset",
        "kaggle_files": "Fake.csv,True.csv",
        "out_subdir": "fake_real",
        "type": "kaggle",
        "title": "Kaggle Fake and Real News",
        "manual_url": "https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset",
    },
    "liar": {
        "direct_url": "https://www.cs.ucsb.edu/~william/data/liar_dataset.zip",
        "out_subdir": "liar",
        "type": "direct",
        "title": "LIAR (UCSB Politifact fact-checks)",
        "manual_url": "https://www.cs.ucsb.edu/~william/data/liar_dataset.zip",
    },
    "fakenewsnet": {
        "github_repo": "https://github.com/KaiDMML/FakeNewsNet",
        "out_subdir": "fakenewsnet",
        "type": "github",
        "title": "FakeNewsNet (KaiDMML)",
        "manual_url": "https://github.com/KaiDMML/FakeNewsNet",
        "mirror_env": "FNN_MIRROR_URL",
    },
    "welfake": {
        "kaggle_slug": "saurabhshahane/fake-news-classification",
        "kaggle_files": "WELFake_Dataset.csv",
        "out_subdir": "welfake",
        "type": "kaggle",
        "title": "WELFake (Kaggle, 72k articles)",
        "manual_url": "https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification",
    },
}


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _println(text: str = "") -> None:
    """Print without throwing on broken-pipe (e.g. when piped to `head`)."""
    try:
        print(text)
    except BrokenPipeError:  # pragma: no cover
        pass


def _section(title: str) -> None:
    """Print a section banner."""
    _println()
    _println("=" * 78)
    _println(f"  {title}")
    _println("=" * 78)


def _have(executable: str) -> bool:
    """Return True if `executable` is on the PATH."""
    return shutil.which(executable) is not None


def _have_kaggle_creds() -> bool:
    """Return True if Kaggle credentials are available in any form.

    Kaggle credentials are checked in this priority order:
      1. ``KAGGLE_USERNAME`` and ``KAGGLE_KEY`` env vars.
      2. ``~/.kaggle/kaggle.json`` (the file `kaggle` CLI writes on `kaggle configure`).
      3. The ``kaggle`` CLI being on the PATH (assumes the user has already
         authenticated via the CLI — the underlying `~/.kaggle/kaggle.json`
         must still exist, but we don't second-guess the CLI).
    """
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if kaggle_json.exists():
        try:
            data = json.loads(kaggle_json.read_text())
            if data.get("username") and data.get("key"):
                return True
        except Exception:  # malformed JSON — keep going
            pass
    return False


def _have_network(host: str = "8.8.8.8", port: int = 53, timeout: float = 3.0) -> bool:
    """Quick, non-fatal network check (does a TCP connect to DNS)."""
    import socket
    try:
        socket.setdefaulttimeout(timeout)
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _download(url: str, dest: Path, timeout: float = 30.0) -> bool:
    """Download `url` to `dest` (file path). Offline-safe.

    Returns True on success, False on any failure (network, HTTP, disk).
    Never raises — the script must always exit 0.
    """
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "fake-news-jev/0.1"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status != 200:
                _println(f"  ! HTTP {resp.status} for {url}")
                return False
            dest.parent.mkdir(parents=True, exist_ok=True)
            with dest.open("wb") as fh:
                while True:
                    chunk = resp.read(64 * 1024)
                    if not chunk:
                        break
                    fh.write(chunk)
        return True
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, TimeoutError) as e:
        _println(f"  ! download failed: {e}")
        return False
    except Exception as e:  # pragma: no cover — defensive last-ditch
        _println(f"  ! unexpected download error: {e}")
        return False


# ---------------------------------------------------------------------------
# Per-dataset downloaders
# ---------------------------------------------------------------------------

def fetch_kaggle(name: str, spec: dict[str, str], out_root: Path) -> bool:
    """Try to download a Kaggle dataset via the `kaggle` CLI; fall back to instructions.

    We prefer the `kaggle` CLI when available because it handles auth,
    pagination and large files correctly. If the CLI is missing OR no
    Kaggle credentials are available, we print the manual steps and return
    False (the script continues; the dataset will simply be unavailable
    for downstream loaders, which fall back to the synthetic generator).
    """
    _section(f"{name} — {spec['title']} (Kaggle)")
    out_dir = out_root / spec["out_subdir"]
    _println(f"  target directory: {out_dir}")
    slug = spec["kaggle_slug"]
    files = spec["kaggle_files"]

    # 1. Fast path: CLI + creds present.
    if _have("kaggle") and _have_kaggle_creds():
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = ["kaggle", "datasets", "download", "-d", slug,
               "-p", str(out_dir), "--unzip"]
        _println(f"  $ {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            _println(f"  ✓ downloaded {name} to {out_dir}")
            return True
        except subprocess.CalledProcessError as e:
            _println(f"  ! kaggle CLI failed: {e.stderr.strip() or e.stdout.strip()}")
            _println("  ! falling back to manual instructions below.")
        except FileNotFoundError:
            _println("  ! kaggle CLI not on PATH; printing manual instructions.")

    # 2. Manual path.
    _println()
    _println("  MANUAL DOWNLOAD STEPS:")
    _println("  -----------------------")
    _println(f"  1. Sign in to Kaggle:      https://www.kaggle.com/")
    _println(f"  2. Dataset URL:            {spec['manual_url']}")
    _println(f"  3. Click 'Download' and unzip the archive.")
    _println(f"  4. Move these files into:  {out_dir}/")
    for fn in files.split(","):
        _println(f"        - {fn.strip()}")
    _println()
    _println("  ALTERNATIVE: install the Kaggle CLI and authenticate:")
    _println("      pip install kaggle")
    _println("      mkdir -p ~/.kaggle && cp <kaggle.json> ~/.kaggle/")
    _println("      chmod 600 ~/.kaggle/kaggle.json")
    _println(f"      kaggle datasets download -d {slug} -p {out_dir} --unzip")
    _println()
    _println("  Set KAGGLE_USERNAME + KAGGLE_KEY env vars for non-interactive runs.")
    return False


def fetch_direct(name: str, spec: dict[str, str], out_root: Path) -> bool:
    """Download a direct-URL dataset (LIAR). Offline-safe."""
    _section(f"{name} — {spec['title']} (direct download)")
    out_dir = out_root / spec["out_subdir"]
    url = spec["direct_url"]
    _println(f"  target directory: {out_dir}")
    _println(f"  source URL:       {url}")

    if not _have_network():
        _println("  ! no network — printing manual steps:")
        _println(f"      curl -L {url} -o {out_dir}/liar_dataset.zip")
        _println(f"      unzip {out_dir}/liar_dataset.zip -d {out_dir}/")
        return False

    zip_path = out_dir / "liar_dataset.zip"
    out_dir.mkdir(parents=True, exist_ok=True)
    ok = _download(url, zip_path)
    if not ok:
        _println("  ! download failed — printing manual steps:")
        _println(f"      curl -L {url} -o {zip_path}")
        _println(f"      unzip {zip_path} -d {out_dir}/")
        return False
    try:
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(out_dir)
        _println(f"  ✓ extracted LIAR TSVs to {out_dir}")
        # Show what landed.
        for p in sorted(out_dir.glob("*.tsv")):
            _println(f"        - {p.name} ({p.stat().st_size:,} bytes)")
        return True
    except (zipfile.BadZipFile, OSError) as e:
        _println(f"  ! unzip failed: {e}")
        _println(f"  ! manually: unzip {zip_path} -d {out_dir}/")
        return False


def fetch_github(name: str, spec: dict[str, str], out_root: Path) -> bool:
    """Print git-clone instructions for FakeNewsNet; honor a community mirror env."""
    _section(f"{name} — {spec['title']} (GitHub)")
    out_dir = out_root / spec["out_subdir"]
    _println(f"  target directory: {out_dir}")
    _println(f"  source repo:      {spec['github_repo']}")
    _println()
    _println("  NOTE: FakeNewsNet ships as per-article JSON files inside the")
    _println("  repo (no packaged download). To get the dataset you must clone")
    _println("  the repo and (optionally) flatten to a CSV the loader accepts:")
    _println()
    _println("      git clone https://github.com/KaiDMML/FakeNewsNet.git " +
             f"{out_dir}/FakeNewsNet")
    _println(f"      cd {out_dir}/FakeNewsNet")
    _println("      ls dataset/    # BuzzFeed + PolitiFact subsets")
    _println()
    _println("  The fakenews.data.load_fakenewsnet() loader expects a pre-flattened")
    _println("  CSV at one of:")
    _println(f"      {out_dir}/fakenewsnet.csv")
    _println(f"      {out_dir}/<any>.csv   # picks first .csv in the dir")
    _println()
    _println("  To flatten from JSON to CSV, see:")
    _println("      scripts/flatten_fakenewsnet.py (community helper)")
    _println()

    mirror = os.environ.get(spec["mirror_env"])
    if mirror:
        _println(f"  {spec['mirror_env']} set; attempting mirror download:")
        _println(f"      {mirror}")
        csv_path = out_dir / "fakenewsnet.csv"
        out_dir.mkdir(parents=True, exist_ok=True)
        if _download(mirror, csv_path):
            _println(f"  ✓ downloaded mirror to {csv_path}")
            return True
        _println("  ! mirror download failed — falling back to manual clone.")
    else:
        _println(f"  (Set {spec['mirror_env']}=<url> to use a community mirror.)")
    return False


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _list_only() -> int:
    """Print a table of all datasets and their URLs; do not download."""
    _println()
    _println("fake-news-jev dataset registry")
    _println("-" * 78)
    for name, spec in DATASETS.items():
        _println(f"{name:14s}  {spec['title']}")
        if spec["type"] == "kaggle":
            _println(f"              Kaggle slug: {spec['kaggle_slug']}")
            _println(f"              URL:          {spec['manual_url']}")
        elif spec["type"] == "direct":
            _println(f"              Direct URL:   {spec['direct_url']}")
        elif spec["type"] == "github":
            _println(f"              GitHub repo:  {spec['github_repo']}")
        _println()
    _println("Run without --list to attempt downloads.")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point. Always returns 0 (offline-safe)."""
    parser = argparse.ArgumentParser(
        prog="download_data",
        description="Download (or print instructions for) the four fake-news datasets.",
    )
    parser.add_argument(
        "--out", type=Path, default=DEFAULT_OUT,
        help=f"Output root directory (default: {DEFAULT_OUT})",
    )
    parser.add_argument(
        "--dataset", choices=list(DATASETS) + ["all"], default="all",
        help="Which dataset to fetch (default: all)",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="Just print the dataset registry and exit (no downloads).",
    )
    parser.add_argument(
        "--no-network-check", action="store_true",
        help="Skip the initial network probe (useful inside sandboxes that block ICMP).",
    )
    args = parser.parse_args(argv)

    if args.list:
        return _list_only()

    out_root: Path = args.out
    out_root.mkdir(parents=True, exist_ok=True)

    _println()
    _println("fake-news-jev dataset downloader")
    _println(f"output root: {out_root}")
    if not args.no_network_check and not _have_network():
        _println()
        _println("! No outbound network detected. Will print manual instructions only.")
    _println(f"kaggle CLI on PATH: {_have('kaggle')}")
    _println(f"kaggle credentials: {_have_kaggle_creds()}")

    targets = (DATASETS if args.dataset == "all"
               else {args.dataset: DATASETS[args.dataset]})

    fetched: list[str] = []
    for name, spec in targets.items():
        try:
            kind = spec["type"]
            if kind == "kaggle":
                ok = fetch_kaggle(name, spec, out_root)
            elif kind == "direct":
                ok = fetch_direct(name, spec, out_root)
            elif kind == "github":
                ok = fetch_github(name, spec, out_root)
            else:
                _println(f"  ! unknown dataset type for {name!r}: {kind}")
                ok = False
            if ok:
                fetched.append(name)
        except Exception as e:  # pragma: no cover — defensive
            _println(f"  ! unexpected error fetching {name}: {e}")

    _section("Summary")
    if fetched:
        _println(f"  Successfully fetched: {', '.join(fetched)}")
    else:
        _println("  No datasets fetched automatically.")
    missing = [n for n in targets if n not in fetched]
    if missing:
        _println(f"  Still need manual setup for: {', '.join(missing)}")
    _println()
    _println("  Tip: data.py::make_synthetic_dataset(n=500, seed=42) provides a")
    _println("  deterministic synthetic dataset for tests/demos before real data is")
    _println("  available. The pipeline can be developed against that while the real")
    _println("  datasets are being downloaded manually.")
    return 0


if __name__ == "__main__":
    # Always exit 0 — offline-safe per the task spec.
    try:
        sys.exit(main())
    except KeyboardInterrupt:  # pragma: no cover
        sys.exit(0)
