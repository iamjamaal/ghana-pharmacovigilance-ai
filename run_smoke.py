"""Wrapper: runs run_dapt --smoke-test and saves output to reports/smoke_out.txt (utf-8)."""
import subprocess, sys, pathlib

out = pathlib.Path(__file__).parent / "reports" / "smoke_out.txt"
script = pathlib.Path(__file__).parent / "scripts" / "run_dapt.py"

with open(out, "w", encoding="utf-8") as f:
    result = subprocess.run(
        [sys.executable, str(script), "--smoke-test"],
        stdout=f, stderr=subprocess.STDOUT,
        cwd=str(pathlib.Path(__file__).parent),
    )

print(f"Done. Exit code {result.returncode}. Output -> {out}")
