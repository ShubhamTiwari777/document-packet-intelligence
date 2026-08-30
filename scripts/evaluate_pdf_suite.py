"""Evaluate the pipeline over every annotated PDF fixture and report the aggregate.

The corpus benchmarks (TABME++, OpenPSS, DocSplit) measure Stage 1 on thousands of page pairs, but
their pages are OCR dumps: TABME++ in particular stores each page as a single line of space-joined
words, so seven of the twenty-one features are degenerate there and no line-based behaviour can be
observed at all. Real PDFs are the only place the full feature set is exercised end to end.

One real PDF is not evidence. This runs every fixture that ships with a ground-truth file, scores
each the same way the API does, and reports a pooled figure across all of them so a single
favourable packet cannot carry the result. Add a PDF and a matching `<name>_ground_truth.json` to
data/samples and it joins the suite automatically.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.evaluation.packet_eval import evaluate_packet
from src.pipeline import DocumentPipeline

parser = argparse.ArgumentParser()
parser.add_argument("--samples", default="data/samples")
parser.add_argument("--config", default="config/default.yaml")
parser.add_argument("--output", default="outputs/benchmarks/pdf_suite.json")
parser.add_argument("--work", default="outputs/pdf_suite")
args = parser.parse_args()


def ground_truth_for(pdf: Path) -> Path | None:
    """A fixture is included when a ground-truth file sits beside it."""
    candidates = [pdf.with_name(f"{pdf.stem}_ground_truth.json")]
    if pdf.stem == "sample_packet":          # the original fixture predates the naming convention
        candidates.append(pdf.with_name("ground_truth.json"))
    return next((c for c in candidates if c.exists()), None)


config = load_config(args.config)
pipeline = DocumentPipeline(config)
work = Path(args.work)

rows, pooled_pred, pooled_true = [], [], []
for pdf in sorted(Path(args.samples).glob("*.pdf")):
    truth_path = ground_truth_for(pdf)
    if not truth_path:
        continue
    truth = json.loads(truth_path.read_text(encoding="utf-8"))
    documents = truth.get("documents", [])
    if len(documents) < 2:      # single-document fixtures cannot exercise boundary detection
        continue

    result = pipeline.process(pdf, work / pdf.stem, packet_id=pdf.stem)
    pages = json.loads((work / pdf.stem / "pages.json").read_text(encoding="utf-8"))
    report = evaluate_packet(result["documents"], documents, len(pages))

    classification = report.get("classification", {})
    rows.append({
        "packet": pdf.name, "pages": len(pages),
        "documents_actual": report["documents_actual"],
        "documents_predicted": report["documents_predicted"],
        "exact_splits": report["documents_exactly_split"],
        "grouping": report["page_grouping_accuracy"],
        "boundary_f1": report["boundary"]["f1"],
        "boundary_precision": report["boundary"]["precision"],
        "boundary_recall": report["boundary"]["recall"],
        "trivial_f1": report["boundary"]["trivial_f1"],
        "lift": report["boundary"]["lift_over_trivial"],
        "type_accuracy": classification.get("accuracy"),
        "type_macro_precision": classification.get("macro_precision"),
    })
    # Pool document-type decisions so the aggregate is over documents, not over packets: a
    # 7-document packet should not weigh the same as a 4-document one.
    for document in report["documents"]:
        pooled_pred.append(document["predicted_type"])
        pooled_true.append(document["actual_type"])

if not rows:
    raise SystemExit("No annotated PDF fixtures found.")

print(f"{'packet':<34}{'pages':>6}{'docs':>6}{'exact':>8}{'grouping':>10}"
      f"{'F1':>7}{'lift':>8}{'type acc':>10}")
for r in rows:
    exact = f"{r['exact_splits']}/{r['documents_actual']}"
    accuracy = r["type_accuracy"] if r["type_accuracy"] is not None else float("nan")
    print(f"{r['packet']:<34}{r['pages']:>6}{r['documents_actual']:>6}{exact:>8}"
          f"{r['grouping']:>10.4f}{r['boundary_f1']:>7.3f}{r['lift']:>+8.3f}{accuracy:>10.3f}")

packets = len(rows)
total_docs = sum(r["documents_actual"] for r in rows)
total_exact = sum(r["exact_splits"] for r in rows)
correct_types = sum(p == a for p, a in zip(pooled_pred, pooled_true))
mean = lambda key: sum(r[key] for r in rows) / packets

print("-" * 89)
pooled_exact = f"{total_exact}/{total_docs}"
print(f"{'POOLED across ' + str(packets) + ' PDFs':<34}{sum(r['pages'] for r in rows):>6}"
      f"{total_docs:>6}{pooled_exact:>8}{mean('grouping'):>10.4f}"
      f"{mean('boundary_f1'):>7.3f}{mean('lift'):>+8.3f}{correct_types/total_docs:>10.3f}")
print(f"\n  documents split at exactly the right pages : {total_exact}/{total_docs} "
      f"({total_exact/total_docs:.1%})")
print(f"  document types identified correctly        : {correct_types}/{total_docs} "
      f"({correct_types/total_docs:.1%})")
print(f"  mean lift over the always-split baseline   : {mean('lift'):+.3f}")

output = Path(args.output)
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps({
    "packets": packets, "documents": total_docs,
    "exact_splits": total_exact, "types_correct": correct_types,
    "mean_grouping": round(mean("grouping"), 4), "mean_lift": round(mean("lift"), 4),
    "per_packet": rows}, indent=2), encoding="utf-8")
print(f"\n-> {output}")
