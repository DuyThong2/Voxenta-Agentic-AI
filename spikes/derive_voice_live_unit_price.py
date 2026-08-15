"""Derive Voice Live's REAL $/second rate from Azure Cost Management, to replace the
OpenAI-borrowed-ratio ESTIMATE in src/config/ai_usage_pricing.py's
DURATION_PRICING_PER_SECOND["azure_voice_live_input"] (see the long comment at that constant).

Why this needs a different approach than every other pricing source in this file: the $/token
UNIT PRICE for Voice Live is already real (Azure Retail Prices API, public, no key -- "Voice Live
API Std - Standard Speech Audio Input Tokens" = $0.015/1K token, confirmed 2026-08-14). What's
NOT real is the tokens-per-second CONVERSION -- the SDK never fires `response.done` (the only
event carrying real token usage) because this project runs Voice Live with
`turn_detection.create_response=False` (STT+VAD only, no model-generated response -- see
infra/voice_live_client.py). So the current constant is entirely a borrowed OpenAI ratio (10
token/sec for realtime audio input), never confirmed against what Azure actually bills THIS
account for THIS traffic pattern.

Azure Cost Management gives the missing piece from the other direction: real $ actually billed,
aggregated by meter over a date range (not per-session, no token counts) -- ONLY at
subscription/resource-group granularity, with a data latency of roughly a day. Divide that by the
real total `duration_ms` this project already recorded for `azure_voice_live_input` in the SAME
window (BE's `ai_usage_record` table, via VOX_BE_DATABASE_URL) and the token-counting question
disappears entirely -- both sides already reflect whatever Azure actually charged for whatever
audio was actually pushed, no need to know the true tokenization/silence-handling behavior.

Auth is a separate control-plane App Registration (AZURE_TENANT_ID/AZURE_CLIENT_ID/
AZURE_CLIENT_SECRET/AZURE_SUBSCRIPTION_ID in .env) with the "Cost Management Reader" role on the
subscription -- NOT the same as AZURE_SPEECH_KEY (that's a data-plane key for calling Voice Live
itself, different auth system entirely, Azure AD app registrations are unrelated to it).

Sample-size caveat (same philosophy as derive_openai_unit_price.py's MIN_TOKENS_TOTAL, adapted):
this project's ai_usage_record data for "azure_voice_live_input" is brand new (added 2026-08-15,
not yet committed) -- expect very little or zero real traffic until students actually run exam/
practice sessions after this lands. The script prints the derived rate regardless, but warns
loudly below MIN_DURATION_SECONDS_TOTAL rather than silently trusting a near-empty sample.

Run from repo root:

    uv run python spikes/derive_voice_live_unit_price.py --days 30
    uv run python spikes/derive_voice_live_unit_price.py --from 2026-08-01 --to 2026-08-15
"""

from __future__ import annotations

import argparse
import os
import socket
from datetime import datetime, timedelta, timezone
from pathlib import Path

import psycopg
import requests
import urllib3.util.connection as urllib3_conn
from dotenv import load_dotenv

# Workaround for this dev machine: TLS handshakes to management.azure.com over IPv6 get reset
# (confirmed with `curl -v` -- connects fine, then "Recv failure: Connection was reset" during the
# handshake), while the same request over IPv4 succeeds cleanly. login.microsoftonline.com (used
# for the token above) is unaffected, only the ARM/Cost Management endpoint. Force urllib3 (what
# `requests` uses under the hood) to resolve IPv4-only so it never picks the broken IPv6 route.
urllib3_conn.allowed_gai_family = lambda: socket.AF_INET

SPIKES_DIR = Path(__file__).resolve().parent
REPO_ROOT = SPIKES_DIR.parent

TOKEN_URL_TMPL = "https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token"
COST_MANAGEMENT_URL_TMPL = (
    "https://management.azure.com/subscriptions/{subscription_id}"
    "/providers/Microsoft.CostManagement/query?api-version=2023-11-01"
)
METER_NAME_SUBSTRING = "voice live"  # case-insensitive match against the "Meter" dimension

MIN_DURATION_SECONDS_TOTAL = 60  # cảnh báo (không chặn) nếu mẫu ai_usage_record ít hơn mức này


def get_management_token(tenant_id: str, client_id: str, client_secret: str) -> str:
    resp = requests.post(
        TOKEN_URL_TMPL.format(tenant_id=tenant_id),
        data={
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
            "scope": "https://management.azure.com/.default",
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


def fetch_cost_rows(token: str, subscription_id: str, start: datetime, end: datetime) -> tuple[list[str], list[list]]:
    """Returns (column_names, rows) from Cost Management, grouped by Meter, daily granularity."""
    body = {
        "type": "ActualCost",
        "timeframe": "Custom",
        "timePeriod": {
            "from": start.strftime("%Y-%m-%dT00:00:00+00:00"),
            "to": end.strftime("%Y-%m-%dT00:00:00+00:00"),
        },
        "dataset": {
            "granularity": "Daily",
            "aggregation": {"totalCost": {"name": "Cost", "function": "Sum"}},
            "grouping": [{"type": "Dimension", "name": "Meter"}],
        },
    }
    resp = requests.post(
        COST_MANAGEMENT_URL_TMPL.format(subscription_id=subscription_id),
        json=body,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        timeout=30,
    )
    resp.raise_for_status()
    props = resp.json()["properties"]
    columns = [c["name"] for c in props["columns"]]
    return columns, props["rows"]


def sum_voice_live_cost(columns: list[str], rows: list[list]) -> tuple[float, str | None]:
    """Sum the Cost column across rows whose Meter contains METER_NAME_SUBSTRING. Returns
    (total_cost, currency) -- currency is None if there were no matching rows at all."""
    cost_idx = columns.index("Cost")
    meter_idx = columns.index("Meter")
    currency_idx = columns.index("Currency") if "Currency" in columns else None

    total = 0.0
    currency: str | None = None
    for row in rows:
        meter = str(row[meter_idx] or "")
        if METER_NAME_SUBSTRING not in meter.lower():
            continue
        total += float(row[cost_idx])
        if currency_idx is not None:
            currency = row[currency_idx]
    return total, currency


def sum_recorded_duration_ms(database_url: str, start: datetime, end: datetime) -> tuple[float, int]:
    """Sum(duration_ms) + row count for provider='azure_voice_live_input' in [start, end)."""
    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT COALESCE(SUM(duration_ms), 0), COUNT(*)
                FROM ai_usage_record
                WHERE provider = 'azure_voice_live_input'
                  AND occurred_at >= %s AND occurred_at < %s
                """,
                (start, end),
            )
            total_duration_ms, row_count = cur.fetchone()
    return float(total_duration_ms), int(row_count)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=30, help="Số ngày nhìn lại nếu không dùng --from/--to (default: 30)")
    parser.add_argument("--from", dest="date_from", default=None, help="YYYY-MM-DD, ghi đè --days")
    parser.add_argument("--to", dest="date_to", default=None, help="YYYY-MM-DD, ghi đè --days (default: hôm nay)")
    args = parser.parse_args()

    load_dotenv(dotenv_path=REPO_ROOT / ".env")

    tenant_id = os.getenv("AZURE_TENANT_ID")
    client_id = os.getenv("AZURE_CLIENT_ID")
    client_secret = os.getenv("AZURE_CLIENT_SECRET")
    subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID")
    be_database_url = os.getenv("VOX_BE_DATABASE_URL")

    missing = [
        name
        for name, value in [
            ("AZURE_TENANT_ID", tenant_id),
            ("AZURE_CLIENT_ID", client_id),
            ("AZURE_CLIENT_SECRET", client_secret),
            ("AZURE_SUBSCRIPTION_ID", subscription_id),
            ("VOX_BE_DATABASE_URL", be_database_url),
        ]
        if not value
    ]
    if missing:
        raise SystemExit(f"Thiếu biến .env: {', '.join(missing)}")

    end = datetime.now(timezone.utc) if not args.date_to else datetime.strptime(args.date_to, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    if args.date_from:
        start = datetime.strptime(args.date_from, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    else:
        start = end - timedelta(days=args.days)

    print(f"Khoảng thời gian: {start.date()} -> {end.date()}")

    token = get_management_token(tenant_id, client_id, client_secret)
    columns, rows = fetch_cost_rows(token, subscription_id, start, end)
    total_cost, currency = sum_voice_live_cost(columns, rows)

    total_duration_ms, row_count = sum_recorded_duration_ms(be_database_url, start, end)
    total_seconds = total_duration_ms / 1000.0

    print(f"\nChi phí Voice Live thật (Cost Management, meter chứa '{METER_NAME_SUBSTRING}'): "
          f"{total_cost:.6f} {currency or '(không có row nào khớp)'}")
    print(f"Tổng duration_ms đã ghi (ai_usage_record, provider='azure_voice_live_input'): "
          f"{total_duration_ms:.0f} ms ({total_seconds:.1f}s, {row_count} row)")

    if currency and currency != "USD":
        print(f"\nCẢNH BÁO: Cost Management trả về đơn vị tiền tệ '{currency}', KHÔNG PHẢI USD -- "
              "phải tự quy đổi trước khi so sánh/cập nhật ai_usage_pricing.py (đang tính bằng USD).")

    if total_seconds < MIN_DURATION_SECONDS_TOTAL:
        print(f"\nCẢNH BÁO: chỉ có {total_seconds:.1f}s dữ liệu (< {MIN_DURATION_SECONDS_TOTAL}s ngưỡng tối "
              "thiểu) -- mẫu quá nhỏ để tin, KHÔNG nên cập nhật ai_usage_pricing.py với số này ngay. "
              "Việc track azure_voice_live_input còn rất mới (2026-08-15) -- cần thêm phiên thi/luyện tập "
              "thật trước khi chạy lại.")
        return

    if total_cost <= 0:
        print("\nKHÔNG có chi phí Voice Live nào trong Cost Management cho khoảng thời gian này -- "
              "có thể do độ trễ dữ liệu (~1 ngày) chưa phản ánh traffic gần đây, hoặc meter name đã đổi. "
              "Thử --days lớn hơn hoặc kiểm tra tay qua Portal (Cost Management + Billing -> Cost Analysis).")
        return

    price_per_second = total_cost / total_seconds
    print(f"\nGiá THẬT suy ra: ${price_per_second:.8f}/giây")
    print(f"So với ước lượng hiện tại (tỷ lệ mượn OpenAI): $0.00015/giây")
    print("\nCập nhật vào src/config/ai_usage_pricing.py:")
    print(f'    "azure_voice_live_input": {price_per_second:.8f},')


if __name__ == "__main__":
    main()
