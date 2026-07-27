from __future__ import annotations

import argparse
import json
from pathlib import Path

from .experiment_tracking import write_json
from .generation_metrics import aggregate_generation_records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate automatic and human/judge generation-evaluation fields."
    )
    parser.add_argument("--input", required=True, help="Generation JSONL with optional evaluation fields.")
    parser.add_argument("--output", help="Summary JSON. Defaults next to the input file.")
    parser.add_argument("--group-by", default="query_type")
    args = parser.parse_args()

    records = load_jsonl(args.input)
    summary = aggregate_generation_records(records, group_by=args.group_by or None)
    output = args.output or str(Path(args.input).with_suffix(".summary.json"))
    write_json(output, {"schema_version": 1, "source": args.input, "metrics": summary})
    print_summary(summary["overall"])
    print(f"output={output}")


def load_jsonl(path: str | Path) -> list[dict]:
    records = []
    with Path(path).open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            records.append(record)
    return records


def print_summary(overall: dict) -> None:
    print(f"queries={overall['queries']}")
    print(f"citation_format_valid_rate={format_metric(overall['citation_format_valid_rate'])}")
    refusal = overall["refusal"]
    print(f"refusal_precision={format_metric(refusal['precision'])}")
    print(f"refusal_recall={format_metric(refusal['recall'])}")
    print(f"refusal_f1={format_metric(refusal['f1'])}")
    for field, values in overall["human_or_judge_scores"].items():
        print(f"{field}={format_metric(values['mean'])} labeled={values['labeled_queries']}")


def format_metric(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.4f}"


if __name__ == "__main__":
    main()
