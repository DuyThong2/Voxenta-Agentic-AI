"""Query Anthropic's Usage & Cost Admin API for the real USD already billed for a model.

This hits `GET /v1/organizations/cost_report` -- Anthropic's own billing ground truth -- as
opposed to the token-count x placeholder-price *estimate* computed locally in
src/infra/message_broker/ai_usage_tracker.py (see src/config/ai_usage_pricing.py).

Needs an **Admin API key** (`sk-ant-admin01-...`), which is different from the regular
`ANTHROPIC_API_KEY` this project uses for inference (src/utils/confidence_utils.py). Create one
in Console -> Settings -> Organization -> Admin API Keys, then add to the root `.env`:

    ANTHROPIC_ADMIN_API_KEY=sk-ant-admin01-...

Run from repo root:

    uv run python spikes/check_claude_cost.py --model claude-sonnet-4-6 --days 30
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

SPIKES_DIR = Path(__file__).resolve().parent
REPO_ROOT = SPIKES_DIR.parent  # .env lives at repo root

COST_REPORT_URL = "https://api.anthropic.com/v1/organizations/cost_report"


def fetch_cost_report(api_key: str, starting_at: str, ending_at: str) -> list[dict]:
    """Fetch every cost-report result row, following pagination via has_more/next_page."""
    all_results: list[dict] = []
    page: str | None = None
    while True:
        params = {
            "starting_at": starting_at,
            "ending_at": ending_at,
            "group_by[]": "description",  # needed to get `model` back on each result row
        }
        if page:
            params["page"] = page
        resp = requests.get(
            COST_REPORT_URL,
            params=params,
            headers={"anthropic-version": "2023-06-01", "x-api-key": api_key},
            timeout=30,
        )
        resp.raise_for_status()
        body = resp.json()
        for bucket in body["data"]:
            all_results.extend(bucket["results"])
        if not body.get("has_more"):
            break
        page = body["next_page"]
    return all_results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="claude-sonnet-4-6",
        help="Model name to filter on (matches the `model` field the API returns)",
    )
    parser.add_argument("--days", type=int, default=30, help="How many days back to look (default: 30)")
    args = parser.parse_args()

    load_dotenv(dotenv_path=REPO_ROOT / ".env")
    api_key = os.getenv("ANTHROPIC_ADMIN_API_KEY")
    if not api_key:
        raise SystemExit(
            "ANTHROPIC_ADMIN_API_KEY chưa được set trong .env.\n"
            "Tạo Admin API key (sk-ant-admin01-...) tại Console -> Settings -> Organization -> "
            "Admin API Keys -- khác với ANTHROPIC_API_KEY thường dùng để gọi model."
        )

    ending_at = datetime.now(timezone.utc)
    starting_at = ending_at - timedelta(days=args.days)

    results = fetch_cost_report(
        api_key,
        starting_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
        ending_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
    )

    matched = [r for r in results if r.get("model") == args.model]
    total_usd = sum(float(r["amount"]) for r in matched) / 100  # amount is in cents

    print(f"Model: {args.model}")
    print(f"Khoảng thời gian: {starting_at.date()} -> {ending_at.date()}")
    print(f"Số dòng chi phí khớp: {len(matched)}")
    print(f"Tổng chi phí thật đã tiêu: ${total_usd:.4f} USD")

    by_type: dict[str, float] = {}
    for r in matched:
        key = r.get("token_type") or r.get("cost_type") or "unknown"
        by_type[key] = by_type.get(key, 0.0) + float(r["amount"]) / 100
    if by_type:
        print("\nChi tiết theo loại token/cost:")
        for key, usd in sorted(by_type.items(), key=lambda kv: -kv[1]):
            print(f"  {key:45s} ${usd:.4f}")


if __name__ == "__main__":
    main()
