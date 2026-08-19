"""Generate a compact, truthful benchmark status section for the technical report."""
from __future__ import annotations
from pathlib import Path
import json

benchmark_dir = Path("outputs/benchmarks")
lines = ["## Benchmark Artifact Status", ""]
for name in ("stage1_results", "stage2_results", "stage3_results"):
    path = benchmark_dir / f"{name}.json"
    lines.append(f"### {name}")
    lines.append("")
    lines.append("```json")
    lines.append(path.read_text(encoding="utf-8") if path.exists() else '"not generated"')
    lines.append("```")
    lines.append("")
Path("outputs").mkdir(exist_ok=True)
Path("outputs/benchmark_status.md").write_text("\n".join(lines), encoding="utf-8")
print("Wrote outputs/benchmark_status.md")
