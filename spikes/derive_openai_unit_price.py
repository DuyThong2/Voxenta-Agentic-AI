"""Derive OpenAI's REAL per-token price from `/v1/organization/costs` -- as opposed to the
list-price guesses in src/config/ai_usage_pricing.py.

Originally this joined the Costs API (dollars) against the separate Usage API (token counts) by
model+day, since the Costs API docs example showed no token count. Turned out unnecessary: a real
call (tested live with this project's actual admin key) shows every `results[]` row already carries
`quantity` + `quantity_unit: "tokens"` alongside `amount` -- e.g.
`{"line_item": "gpt-5.4-2026-03-05, input", "amount": {"value": "0.0652...", "currency": "usd"},
"quantity": 26099.0, "quantity_unit": "tokens"}`. So `amount.value / quantity` on a SINGLE endpoint
already gives the real $/token OpenAI billed -- no cross-API join, no date/model matching bugs.

Two things the live call also caught, both wrong in earlier drafts of this script:
- `amount.value` is a STRING (sometimes in scientific notation like `"0E-6176"` for an exact-zero
  row) -- must go through `float()`, not used directly as a number.
- The API reports the DATED snapshot name (`gpt-5.4-2026-03-05`), never the bare alias
  (`gpt-5.4`) this project's call sites request -- `--model gpt-5.4` matches by prefix so it keeps
  working across snapshot rollovers.

Needs an **Admin API key** (`sk-admin-...`), different from the regular `OPENAI_API_KEY` this
project uses for inference. Only an organization Owner can create one, at
platform.openai.com -> Settings -> Organization -> Admin keys, scope `api.usage.read`. Add to the
root `.env` as a SEPARATE secret (do not reuse OPENAI_API_KEY -- an admin key can read
organization-wide financial data, different risk/blast-radius than an inference key):

    OPENAI_ADMIN_KEY=sk-admin-...

Two guards applied before trusting a derived price (both requested explicitly, see plan doc):
1. A (model, token-category) is only trusted if its TOTAL tokens across the whole period reach
   MIN_TOKENS_TOTAL -- otherwise kept as "not enough data". This is a period-level guard, not a
   per-day one: the price is computed by summing $ and tokens across every day first and dividing
   ONCE at the end (not by averaging a separate price computed per day), so a single low-volume day
   can't skew the result any more than its actual share of the total warrants -- the risk that
   needed guarding against was always "total sample too small", not "any one day too small". A
   per-day threshold was tried first and it discarded real signal for low-but-steady-traffic models
   (e.g. gpt-4o-mini: 9 real days, none individually reaching 10k tokens/day, but 16,751 tokens
   total -- a per-day gate reported "no data" for a model that in fact had plenty of it).
2. Cached input tokens are tracked and divided SEPARATELY from uncached input tokens (the API
   already gives them as separate line_items, "input" vs "cached input") -- summing them together
   would understate the real input price (cached tokens are ~90% cheaper, so a model with a high
   cache-hit rate would otherwise look artificially cheap for ALL its input, not just the cached
   part).

Run from repo root:

    uv run python spikes/derive_openai_unit_price.py --model gpt-5.4 --days 30
"""

from __future__ import annotations

import argparse
import os
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

SPIKES_DIR = Path(__file__).resolve().parent
REPO_ROOT = SPIKES_DIR.parent

COSTS_URL = "https://api.openai.com/v1/organization/costs"

MIN_TOKENS_TOTAL = 10_000  # bỏ qua (model, category) nào có TỔNG token cả period ít hơn mức này -- xem docstring


def fetch_cost_buckets(api_key: str, start_time: int, end_time: int) -> list[dict]:
    """Fetch every time bucket from the Costs API, following pagination via has_more/next_page."""
    buckets: list[dict] = []
    page: str | None = None
    while True:
        params: dict[str, object] = {
            "start_time": start_time,
            "end_time": end_time,
            "bucket_width": "1d",
            "group_by[]": ["line_item"],
            "limit": 31,
        }
        if page:
            params["page"] = page
        resp = requests.get(
            COSTS_URL,
            params=params,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            timeout=30,
        )
        resp.raise_for_status()
        body = resp.json()
        buckets.extend(body.get("data", []))
        if not body.get("has_more"):
            break
        page = body["next_page"]
    return buckets


def _bucket_date(bucket: dict) -> str:
    return datetime.fromtimestamp(bucket["start_time"], tz=timezone.utc).date().isoformat()


def _matches_model(actual_model: str, alias: str) -> bool:
    """OpenAI's Costs API reports the dated snapshot ("gpt-5.4-2026-03-05"), not the bare alias
    ("gpt-5.4") this project's call sites request (confirmed live: ChatOpenAI(model="gpt-5.4")
    resolves to that snapshot server-side). Match on prefix so --model gpt-5.4 keeps working across
    snapshot rollovers, without requiring the caller to know today's exact dated name."""
    return actual_model == alias or actual_model.startswith(alias + "-")


def _categorize_line_item(line_item: str, model: str) -> str | None:
    """"gpt-5.4-2026-03-05, input" -> "input"; "..., cached input" -> "cached"; None if not this model."""
    head, _, rest = line_item.partition(",")
    if not _matches_model(head.strip(), model):
        return None
    rest = rest.strip().lower()
    if "cached" in rest:
        return "cached"
    if "output" in rest:
        return "output"
    if "input" in rest:
        return "input"
    return None


def collect_usd_and_tokens(cost_buckets: list[dict], model: str) -> tuple[dict, dict]:
    """{date: {category: usd}}, {date: {category: tokens}} for this model."""
    usd_by_day: dict[str, dict[str, float]] = defaultdict(lambda: {"input": 0.0, "output": 0.0, "cached": 0.0})
    tokens_by_day: dict[str, dict[str, int]] = defaultdict(lambda: {"input": 0, "output": 0, "cached": 0})
    for bucket in cost_buckets:
        date = _bucket_date(bucket)
        for row in bucket.get("results", []):
            line_item = row.get("line_item") or ""
            category = _categorize_line_item(line_item, model)
            if category is None:
                continue
            usd_by_day[date][category] += float(row["amount"]["value"])
            tokens_by_day[date][category] += int(row.get("quantity") or 0)
    return usd_by_day, tokens_by_day


def derive_price_per_mtok(
    usd_by_day: dict[str, dict[str, float]],
    tokens_by_day: dict[str, dict[str, int]],
) -> dict[str, float | None]:
    # Cộng dồn $ và token qua TOÀN BỘ ngày trước, ngưỡng đủ-mẫu chỉ áp dụng ở dòng cuối trên tổng
    # đã cộng dồn -- không lọc/bỏ ngày nào giữa chừng, vì phép chia chỉ xảy ra MỘT LẦN ở cuối
    # (xem docstring: đây là weighted average theo khối lượng, không phải trung bình của các giá
    # tính riêng từng ngày, nên ngưỡng "đủ mẫu" phải đo trên tổng, không phải từng ngày).
    totals_usd = {"input": 0.0, "output": 0.0, "cached": 0.0}
    totals_tokens = {"input": 0, "output": 0, "cached": 0}

    for tokens in tokens_by_day.values():
        for category in ("input", "output", "cached"):
            totals_tokens[category] += tokens[category]
    for usd in usd_by_day.values():
        for category in ("input", "output", "cached"):
            totals_usd[category] += usd[category]

    not_enough_data = [c for c in ("input", "output", "cached") if totals_tokens[c] < MIN_TOKENS_TOTAL]
    if not_enough_data:
        print("Không đủ dữ liệu (tổng < %d token cả period):" % MIN_TOKENS_TOTAL)
        for category in not_enough_data:
            print(f"  {category}: tổng {totals_tokens[category]} token")

    derived: dict[str, float | None] = {}
    for category in ("input", "output", "cached"):
        if category in not_enough_data:
            derived[category] = None
        else:
            derived[category] = (totals_usd[category] / totals_tokens[category]) * 1_000_000
    return derived


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Tên model đúng như OpenAI trả về (vd gpt-5.4)")
    parser.add_argument("--days", type=int, default=30, help="Số ngày nhìn lại (default: 30)")
    args = parser.parse_args()

    load_dotenv(dotenv_path=REPO_ROOT / ".env")
    api_key = os.getenv("OPENAI_ADMIN_KEY")
    if not api_key:
        raise SystemExit(
            "OPENAI_ADMIN_KEY chưa được set trong .env.\n"
            "Tạo Admin key (sk-admin-..., scope api.usage.read) tại platform.openai.com -> "
            "Settings -> Organization -> Admin keys -- chỉ Owner mới tạo được, và đây PHẢI là "
            "secret riêng, không dùng chung với OPENAI_API_KEY."
        )

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=args.days)
    start_ts, end_ts = int(start.timestamp()), int(end.timestamp())

    cost_buckets = fetch_cost_buckets(api_key, start_ts, end_ts)
    usd_by_day, tokens_by_day = collect_usd_and_tokens(cost_buckets, args.model)

    if not tokens_by_day:
        raise SystemExit(f"Không có chi phí nào cho model '{args.model}' trong {args.days} ngày qua.")

    derived = derive_price_per_mtok(usd_by_day, tokens_by_day)

    print(f"\nModel: {args.model}")
    print(f"Khoảng thời gian: {start.date()} -> {end.date()}")
    print("\nĐơn giá THẬT suy ra ($/1M token):")
    for category, price in derived.items():
        label = {"input": "input (uncached)", "output": "output", "cached": "cached input"}[category]
        if price is None:
            print(f"  {label:20s} không đủ dữ liệu (mẫu quá nhỏ hoặc chưa dùng)")
        else:
            print(f"  {label:20s} ${price:.4f}")

    print(
        "\nSo sánh với src/config/ai_usage_pricing.py trước khi cập nhật LLM_PRICING -- "
        "số ở đây LÀ giá thật đã calibrate, khác với cached_input_per_mtok=input*0.10 đang "
        "để tạm."
    )


if __name__ == "__main__":
    main()
