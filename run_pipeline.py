#!/usr/bin/env python3
"""
Ghana ADR Dataset Pipeline — Main Runner
=========================================
Builds a unified NLP dataset from Ghanaian pharmacovigilance sources.

Usage:
    python run_pipeline.py          # Run all steps
    python run_pipeline.py --step 2 # Run from step 2 onwards
    python run_pipeline.py --only 3 # Run only step 3

Steps:
    1. Download sources (DrugLens PDFs + PMC articles)
    2. Extract text (PDF → text, XML → sections)
    3. Segment & auto-annotate (sentences + lexicon matching)
    4. Export (JSONL, CSV, NER, RE, Label Studio, dataset card)
"""

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent / "scripts"

STEPS = [
    ("01_download.py", "Download Ghanaian data sources"),
    ("02_extract.py", "Extract text from PDFs and PMC XML"),
    ("03_segment_annotate.py", "Segment into sentences & auto-annotate"),
    ("04_export.py", "Export unified dataset in multiple formats"),
]


def run_step(script_name: str, description: str, step_num: int) -> bool:
    print(f"\n{'='*60}")
    print(f"  STEP {step_num}: {description}")
    print(f"  Script: {script_name}")
    print(f"{'='*60}\n")

    script_path = SCRIPTS_DIR / script_name
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(script_path.parent.parent)
    )

    if result.returncode != 0:
        print(f"\n  ✗ Step {step_num} failed with exit code {result.returncode}")
        return False

    print(f"\n  ✓ Step {step_num} complete")
    return True


def main():
    parser = argparse.ArgumentParser(description="Ghana ADR Dataset Pipeline")
    parser.add_argument("--step", type=int, default=1,
                        help="Start from this step (default: 1)")
    parser.add_argument("--only", type=int, default=None,
                        help="Run only this step")
    args = parser.parse_args()

    print("╔══════════════════════════════════════════════════════════╗")
    print("║  Ghana ADR NLP Dataset Pipeline                        ║")
    print("║  Ghana AI Innovation Challenge 2026                    ║")
    print("╚══════════════════════════════════════════════════════════╝")

    if args.only:
        if 1 <= args.only <= len(STEPS):
            script, desc = STEPS[args.only - 1]
            success = run_step(script, desc, args.only)
            sys.exit(0 if success else 1)
        else:
            print(f"Invalid step: {args.only}. Choose 1-{len(STEPS)}")
            sys.exit(1)

    for i, (script, desc) in enumerate(STEPS, 1):
        if i < args.step:
            continue
        success = run_step(script, desc, i)
        if not success:
            print(f"\nPipeline halted at step {i}. Fix the error and re-run with --step {i}")
            sys.exit(1)

    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║  Pipeline complete!                                    ║")
    print("║                                                        ║")
    print("║  Outputs in: ./output/                                 ║")
    print("║    • ghana_adr_dataset.jsonl  (full dataset)           ║")
    print("║    • ghana_adr_dataset.csv    (spreadsheet view)       ║")
    print("║    • ghana_adr_ner.jsonl      (NER training format)    ║")
    print("║    • ghana_adr_re.jsonl       (relation extraction)    ║")
    print("║    • label_studio_import.json (manual review)          ║")
    print("║    • dataset_card.md          (HuggingFace card)       ║")
    print("╚══════════════════════════════════════════════════════════╝")


if __name__ == "__main__":
    main()
