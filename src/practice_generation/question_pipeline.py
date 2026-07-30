"""Offline entry point for the question-generation graph."""

import argparse
import json
import os
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

from node.questionGenerationGraph.pipeline import GenerationPipeline

load_dotenv(Path(__file__).resolve().parents[2] / ".env")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("seed-output") / date.today().isoformat(),
    )
    parser.add_argument("--persist", action="store_true")
    parser.add_argument("--dsn", default=os.getenv("VOX_DB_DSN"))
    args = parser.parse_args()
    pipeline = GenerationPipeline(
        output_root=args.output,
        persist=args.persist,
        dsn=args.dsn,
    )
    summary = pipeline.run()
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
