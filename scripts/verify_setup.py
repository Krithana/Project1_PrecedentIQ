"""
Run this after every setup step to confirm things actually worked.

Usage:
    python scripts/verify_setup.py
"""
import importlib
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

REQUIRED_PACKAGES = [
    "boto3", "datasets", "huggingface_hub", "pymupdf", "pdfplumber",
    "sentence_transformers", "transformers", "torch", "qdrant_client",
    "rank_bm25", "langchain", "langchain_community", "ollama",
    "fastapi", "uvicorn", "streamlit", "pydantic", "docx", "tqdm", "yaml",
]

REQUIRED_DIRS = [
    "data/raw", "data/processed", "data/validation", "config", "scripts", "src",
]


def check_packages():
    print("=== 1. Package imports ===")
    ok, failed = [], []
    for pkg in REQUIRED_PACKAGES:
        try:
            importlib.import_module(pkg)
            ok.append(pkg)
        except ImportError as e:
            failed.append((pkg, str(e)))
    print(f"  OK: {len(ok)}/{len(REQUIRED_PACKAGES)}")
    for pkg, err in failed:
        print(f"  [MISSING] {pkg} -> pip install {pkg}   ({err})")
    return len(failed) == 0


def check_dirs():
    print("\n=== 2. Folder structure ===")
    all_ok = True
    for d in REQUIRED_DIRS:
        path = os.path.join(HERE, d)
        exists = os.path.isdir(path)
        print(f"  {'OK' if exists else 'MISSING'}: {d}")
        all_ok = all_ok and exists
    return all_ok


def check_network():
    # A raw TCP connect on port 443 can succeed even when a proxy/firewall blocks
    # the actual HTTP request, so we do a real GET here rather than just a socket handshake.
    print("\n=== 3. Network reachability (real HTTP check) ===")
    import urllib.request
    import urllib.error
    targets = {
        "AWS S3 (judgment source)": "https://indian-supreme-court-judgments.s3.amazonaws.com/?list-type=2&max-keys=1",
        "Hugging Face (validation sets)": "https://huggingface.co",
    }
    all_ok = True
    for label, url in targets.items():
        try:
            with urllib.request.urlopen(url, timeout=8) as resp:
                print(f"  OK: {label} -> HTTP {resp.status}")
        except urllib.error.HTTPError as e:
            body = e.read(200).decode(errors="ignore")
            print(f"  [BLOCKED] {label} -> HTTP {e.code} ({body[:100]})")
            all_ok = False
        except Exception as e:
            print(f"  [UNREACHABLE] {label} -> {e}")
            print("     (If this is a college/office network, try mobile hotspot or a different network.)")
            all_ok = False
    return all_ok


def check_downloaded_data():
    print("\n=== 4. Downloaded data on disk ===")
    raw_dir = os.path.join(HERE, "data", "raw")
    val_dir = os.path.join(HERE, "data", "validation")

    def count_files(d):
        if not os.path.isdir(d):
            return 0
        return sum(len(files) for _, _, files in os.walk(d))

    sc_count = count_files(os.path.join(raw_dir, "supreme_court"))
    hc_count = count_files(os.path.join(raw_dir, "high_court"))
    val_count = count_files(val_dir)
    print(f"  Supreme Court files downloaded: {sc_count}")
    print(f"  High Court files downloaded:    {hc_count}")
    print(f"  Validation set files:           {val_count}")
    if sc_count == 0 and hc_count == 0:
        print("  -> Nothing downloaded yet. Run: python scripts/download_judgments.py")
    if val_count == 0:
        print("  -> No validation sets yet. Run: python scripts/download_validation_sets.py")


def main():
    pkg_ok = check_packages()
    dir_ok = check_dirs()
    net_ok = check_network()
    check_downloaded_data()

    print("\n=== SUMMARY ===")
    print(f"  Packages:  {'PASS' if pkg_ok else 'FAIL — install missing packages above'}")
    print(f"  Folders:   {'PASS' if dir_ok else 'FAIL — re-run project scaffold setup'}")
    print(f"  Network:   {'PASS' if net_ok else 'FAIL — check your internet connection / firewall'}")

    if pkg_ok and dir_ok and net_ok:
        print("\nEverything looks good. Proceed to:")
        print("  python scripts/download_judgments.py")
        print("  python scripts/download_validation_sets.py")
    else:
        print("\nFix the FAILs above before downloading data.")
        sys.exit(1)


if __name__ == "__main__":
    main()
