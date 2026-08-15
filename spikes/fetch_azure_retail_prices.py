"""Pull the REAL Azure Speech-to-Text price for this project's region from Azure's public Retail
Prices API, to replace the guessed `0.006`/second constant in
src/config/ai_usage_pricing.py's DURATION_PRICING_PER_SECOND["azure_stt"].

No auth needed -- https://prices.azure.com/api/retail/prices is public, unauthenticated, no key/
subscription required (confirmed live). Only covers the standalone Azure STT used for the
followup-decision/pronunciation fallback paths (ai_usage_tracker.record_duration_usage("azure_stt",
...)) -- NOT Voice Live (that's billed per audio TOKEN under a completely different meter family,
"Voice Live API Std/Pro/Lite - ... Tokens", see spikes/README.md and the plan doc; Voice Live isn't
tracked in ai_usage_tracker at all yet, out of scope for this script).

The pay-as-you-go (non-Commitment-Tier, non-Custom) STT meter is named "S1 Speech To Text",
billed "1 Hour" -- confirmed by listing every "Speech To Text" meter in the region and filtering
out "Commitment"/"Custom" variants (those are bulk-purchase tiers, not the per-call rate this
project actually pays under PAYG).

Run from repo root:

    uv run python spikes/fetch_azure_retail_prices.py
    uv run python spikes/fetch_azure_retail_prices.py --region eastus  # override region
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import requests
from dotenv import load_dotenv

SPIKES_DIR = Path(__file__).resolve().parent
REPO_ROOT = SPIKES_DIR.parent

RETAIL_PRICES_URL = "https://prices.azure.com/api/retail/prices"
STT_METER_NAME = "S1 Speech To Text"


def fetch_stt_price_per_hour(region: str) -> float:
    """Return the PAYG $/hour rate for standard (non-custom) Azure Speech-to-Text in `region`."""
    resp = requests.get(
        RETAIL_PRICES_URL,
        params={
            "$filter": f"armRegionName eq '{region}' and meterName eq '{STT_METER_NAME}'",
        },
        timeout=30,
    )
    resp.raise_for_status()
    items = resp.json().get("Items", [])
    if not items:
        raise SystemExit(
            f"Không tìm thấy meter '{STT_METER_NAME}' cho region '{region}' -- Azure có thể đã đổi "
            "tên meter, hoặc region này không có STT PAYG. Thử liệt kê thủ công:\n"
            f"  curl \"{RETAIL_PRICES_URL}?$filter=armRegionName eq '{region}' and "
            "contains(meterName,'Speech To Text')\""
        )
    return items[0]["retailPrice"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--region",
        default=None,
        help="Azure region (armRegionName). Default: AZURE_SPEECH_REGION từ .env ở repo root.",
    )
    args = parser.parse_args()

    load_dotenv(dotenv_path=REPO_ROOT / ".env")
    region = args.region or os.getenv("AZURE_SPEECH_REGION")
    if not region:
        raise SystemExit(
            "Không có region -- truyền --region hoặc set AZURE_SPEECH_REGION trong .env."
        )

    price_per_hour = fetch_stt_price_per_hour(region)
    price_per_second = price_per_hour / 3600

    print(f"Meter: {STT_METER_NAME}")
    print(f"Region: {region}")
    print(f"Giá thật: ${price_per_hour:.4f}/giờ  ->  ${price_per_second:.8f}/giây")
    print()
    print("Cập nhật vào src/config/ai_usage_pricing.py:")
    print(f'    "azure_stt": {price_per_second:.8f},')


if __name__ == "__main__":
    main()
