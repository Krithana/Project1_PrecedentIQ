"""
Download validation/benchmark datasets from Hugging Face.

CONFIRMED working (verified against the live dataset pages):
  - IL-TUR / CJPE task  -> holding extraction validation (56 expert-annotated judgments)
  - IL-PCSR             -> precedent retrieval validation (directly matches PrecedentIQ's
                            retrieval task — queries + precedent candidate pool)

NOT confirmed as an installable HF dataset (handled as manual-download fallback below):
  - LawSum / ILDC — these papers exist, but no verified `load_dataset(...)` path was found.
    Use the LexSumm collection instead (see FALLBACK section) which packages a comparable
    Indian Supreme Court summarization dataset (InAbs) in a clean, documented format.

No Hugging Face account needed for any of the confirmed datasets below.

Run:
    python scripts/download_validation_sets.py
"""
import os
import yaml
import truststore

truststore.inject_into_ssl()
from datasets import load_dataset

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_config():
    with open(os.path.join(HERE, "config", "config.yaml")) as f:
        return yaml.safe_load(f)


def main():
    cfg = load_config()
    out_dir = os.path.join(HERE, cfg["paths"]["validation_dir"])
    os.makedirs(out_dir, exist_ok=True)

    # --- IL-TUR: CJPE task (holding/ratio extraction validation) ---
    print("=== IL-TUR — CJPE task (56 expert-annotated judgments) ===")
    try:
        # NOTE: revision="script" is required by this dataset's loader.
        il_tur = load_dataset(
            "Exploration-Lab/IL-TUR",
            cfg["huggingface"]["il_tur_task"],
            revision="script",
            trust_remote_code=True,
        )
        il_tur.save_to_disk(os.path.join(out_dir, "il_tur_cjpe"))
        print(f"  Saved: {il_tur}")
    except Exception as e:
        print(f"  [ERROR] IL-TUR download failed: {e}")
        print("  -> Open huggingface.co/datasets/Exploration-Lab/IL-TUR and confirm the exact")
        print("     task name in the 'Available tasks' list, then update this script.")

    # --- IL-PCSR: precedent retrieval validation (queries + candidate pool) ---
    if not cfg["huggingface"]["pull_il_pcsr"]:
        return
    print("\n=== IL-PCSR — precedent retrieval validation ===")
    try:
        queries = load_dataset("Exploration-Lab/IL-PCSR", name="queries")
        precedents = load_dataset("Exploration-Lab/IL-PCSR", name="precedents")
        queries.save_to_disk(os.path.join(out_dir, "il_pcsr_queries"))
        precedents.save_to_disk(os.path.join(out_dir, "il_pcsr_precedents"))
        print(f"  Saved queries: {queries}")
        print(f"  Saved precedent pool: {precedents}")
    except Exception as e:
        print(f"  [ERROR] IL-PCSR download failed: {e}")
        print("  -> Check huggingface.co/datasets/Exploration-Lab/IL-PCSR for the current API.")

    # --- FALLBACK: LawSum / ILDC could not be confirmed as installable HF datasets ---
    print("\n=== LawSum / ILDC — manual download required ===")
    print("  These were NOT found as a working `load_dataset(...)` path. Recommended fallback:")
    print("  1. LexSumm collection (includes InAbs: Indian SC judgments + headnote summaries)")
    print("     -> github.com/TUMLegalTech/LexSumm-LexT5  (documented, direct download link)")
    print("  2. If you specifically need LawSum or ILDC, check the GitHub links in:")
    print("     - Parikh et al. 2021, arXiv:2110.01188 (LawSum)")
    print("     - Malik et al. 2021, ACL Anthology (ILDC)")
    print("     and place the downloaded files manually in data/validation/<name>/")

    print("\nDone. Run `python scripts/verify_setup.py` next to confirm what landed on disk.")


if __name__ == "__main__":
    main()
