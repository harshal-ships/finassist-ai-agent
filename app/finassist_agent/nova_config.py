"""Nova Sonic region + model resolution (APAC + EU geo-routing)."""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

# Nova 2 Sonic in-region endpoints (AWS Bedrock)
NOVA2_SONIC_REGIONS = frozenset({"us-east-1", "us-west-2", "eu-north-1", "ap-northeast-1"})

# Nova Sonic v1 in-region endpoints (fallback)
NOVA_SONIC_V1_REGIONS = frozenset({"us-east-1", "eu-north-1", "ap-northeast-1"})

APAC_NOVA2_REGION = "ap-northeast-1"
EU_NOVA2_REGION = "eu-north-1"
US_NOVA2_REGION = "us-east-1"


def _requested_region() -> str:
    return (
        os.getenv("NOVA_SONIC_REGION")
        or os.getenv("AWS_DEFAULT_REGION")
        or os.getenv("AWS_REGION")
        or APAC_NOVA2_REGION
    ).strip()


def _route_unsupported_region(requested: str, *, is_nova2: bool) -> str:
    """Pick closest supported Nova endpoint from caller/app region."""
    if requested.startswith("eu-"):
        region = EU_NOVA2_REGION if is_nova2 else EU_NOVA2_REGION
        logger.info(
            "Nova Sonic unavailable in %s — routing to %s (EU)",
            requested,
            region,
        )
        return region

    if requested.startswith("ap-"):
        region = APAC_NOVA2_REGION
        logger.info(
            "Nova Sonic unavailable in %s — routing to %s (APAC)",
            requested,
            region,
        )
        return region

    if requested.startswith("us-"):
        region = US_NOVA2_REGION
        logger.info(
            "Nova Sonic unavailable in %s — routing to %s (US)",
            requested,
            region,
        )
        return region

    logger.warning(
        "Nova Sonic unavailable in %s — falling back to %s",
        requested,
        APAC_NOVA2_REGION,
    )
    return APAC_NOVA2_REGION


def resolve_nova_settings() -> tuple[str, str]:
    """
    Return (region, model_id).

    Explicit override: NOVA_SONIC_REGION=ap-northeast-1 (APAC) or eu-north-1 (EU).
    If unset, auto-routes from AWS_DEFAULT_REGION:
      - ap-southeast-* / ap-*  → ap-northeast-1 (Tokyo)
      - eu-*                   → eu-north-1 (Stockholm)
      - us-*                   → us-east-1
    """
    requested = _requested_region()
    model_id = os.getenv("NOVA_SONIC_MODEL_ID", "amazon.nova-2-sonic-v1:0")
    is_nova2 = "nova-2-sonic" in model_id.lower()

    supported = NOVA2_SONIC_REGIONS if is_nova2 else NOVA_SONIC_V1_REGIONS
    region = requested if requested in supported else _route_unsupported_region(requested, is_nova2=is_nova2)

    return region, model_id
