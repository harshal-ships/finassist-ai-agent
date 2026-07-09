"""Call transcript collection and S3 persistence for Nova Sonic sessions."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

TRANSCRIPT_S3_BUCKET = os.getenv("TRANSCRIPT_S3_BUCKET", "").strip()
TRANSCRIPT_S3_PREFIX = os.getenv("TRANSCRIPT_S3_PREFIX", "finassist/transcripts").strip("/")

# Synthetic USER text injected to trigger the opening greeting — not caller speech.
_INTERNAL_USER_TRIGGERS = frozenset(
    {
        "The call just connected. Deliver your opening greeting now.",
    }
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_interruption_marker(text: str) -> bool:
    return '{ "interrupted" : true }' in text or (
        '"interrupted"' in text and "true" in text
    )


def _is_internal_trigger(text: str) -> bool:
    cleaned = text.strip()
    return cleaned in _INTERNAL_USER_TRIGGERS


def _parse_generation_stage(content_start: dict[str, Any]) -> str | None:
    raw = content_start.get("additionalModelFields")
    if not raw:
        return None
    try:
        fields = json.loads(raw) if isinstance(raw, str) else raw
    except (json.JSONDecodeError, TypeError):
        return None
    stage = fields.get("generationStage")
    return str(stage) if stage else None


@dataclass(frozen=True)
class TranscriptTurn:
    role: str
    content: str
    timestamp: str
    generation_stage: str | None = None
    content_id: str | None = None


@dataclass
class _PendingTurn:
    role: str
    content_id: str
    parts: list[str] = field(default_factory=list)
    generation_stage: str | None = None
    started_at: str = field(default_factory=_utc_now_iso)


class TranscriptCollector:
    """Accumulate USER/ASSISTANT turns from Nova Sonic output events.

    Follows AWS guidance to persist FINAL assistant transcripts (not SPECULATIVE
    previews) and to assemble turns from contentStart → textOutput → contentEnd.
    See: https://docs.aws.amazon.com/nova/latest/nova2-userguide/sonic-chat-history.html
    """

    def __init__(self) -> None:
        self.turns: list[TranscriptTurn] = []
        self._pending: dict[str, _PendingTurn] = {}

    def process_event(self, event: dict[str, Any]) -> None:
        if "contentStart" in event:
            self._on_content_start(event["contentStart"])
        elif "textOutput" in event:
            self._on_text_output(event["textOutput"])
        elif "contentEnd" in event:
            self._on_content_end(event["contentEnd"])

    def _on_content_start(self, start: dict[str, Any]) -> None:
        if start.get("type") not in (None, "TEXT"):
            return
        if "textOutputConfiguration" not in start and start.get("type") != "TEXT":
            return

        content_id = start.get("contentId") or start.get("contentName")
        if not content_id:
            return

        role = str(start.get("role", "ASSISTANT")).upper()
        self._pending[str(content_id)] = _PendingTurn(
            role=role,
            content_id=str(content_id),
            generation_stage=_parse_generation_stage(start),
            started_at=_utc_now_iso(),
        )

    def _on_text_output(self, text_output: dict[str, Any]) -> None:
        text = text_output.get("content", "")
        if not text or _is_interruption_marker(text) or _is_internal_trigger(text):
            return

        content_id = text_output.get("contentId") or text_output.get("contentName")
        role = str(text_output.get("role", "ASSISTANT")).upper()

        if content_id:
            pending = self._pending.get(str(content_id))
            if pending is not None:
                pending.parts.append(text)
                return

        if role == "ASSISTANT" and _parse_generation_stage({"additionalModelFields": text_output.get("additionalModelFields")}) == "SPECULATIVE":
            return

        cleaned = text.strip()
        if cleaned:
            self.turns.append(
                TranscriptTurn(
                    role=role,
                    content=cleaned,
                    timestamp=_utc_now_iso(),
                    content_id=str(content_id) if content_id else None,
                )
            )

    def _on_content_end(self, end: dict[str, Any]) -> None:
        if end.get("type") not in (None, "TEXT"):
            return

        content_id = end.get("contentId") or end.get("contentName")
        if not content_id:
            return

        pending = self._pending.pop(str(content_id), None)
        if pending is None:
            return

        if pending.generation_stage == "SPECULATIVE":
            return

        content = "".join(pending.parts).strip()
        if not content or _is_internal_trigger(content):
            return

        self.turns.append(
            TranscriptTurn(
                role=pending.role,
                content=content,
                timestamp=pending.started_at,
                generation_stage=pending.generation_stage,
                content_id=pending.content_id,
            )
        )

    def flush_pending(self) -> None:
        """Finalize any in-flight turns (e.g. when the call ends abruptly)."""
        for content_id in list(self._pending):
            self._on_content_end({"contentId": content_id, "type": "TEXT"})


@dataclass(frozen=True)
class CallTranscript:
    call_id: str
    caller_number: str
    started_at: str
    ended_at: str
    duration_seconds: float
    agent: str
    platform: str
    nova_model: str
    nova_region: str
    turns: list[TranscriptTurn]

    def to_dict(self) -> dict[str, Any]:
        return {
            "call_id": self.call_id,
            "caller_number": self.caller_number,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration_seconds": round(self.duration_seconds, 3),
            "agent": self.agent,
            "platform": self.platform,
            "nova_model": self.nova_model,
            "nova_region": self.nova_region,
            "turn_count": len(self.turns),
            "turns": [
                {
                    "role": turn.role,
                    "content": turn.content,
                    "timestamp": turn.timestamp,
                    **({"generation_stage": turn.generation_stage} if turn.generation_stage else {}),
                }
                for turn in self.turns
            ],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    @classmethod
    def build(
        cls,
        *,
        call_id: str,
        caller_number: str,
        started_at: float,
        ended_at: float,
        collector: TranscriptCollector,
        agent: str,
        platform: str,
        nova_model: str,
        nova_region: str,
    ) -> CallTranscript:
        return cls(
            call_id=call_id,
            caller_number=caller_number,
            started_at=datetime.fromtimestamp(started_at, tz=timezone.utc).isoformat(),
            ended_at=datetime.fromtimestamp(ended_at, tz=timezone.utc).isoformat(),
            duration_seconds=max(0.0, ended_at - started_at),
            agent=agent,
            platform=platform,
            nova_model=nova_model,
            nova_region=nova_region,
            turns=list(collector.turns),
        )


def transcript_s3_key(call_id: str, ended_at: float) -> str:
    ended = datetime.fromtimestamp(ended_at, tz=timezone.utc)
    date_path = ended.strftime("%Y/%m/%d")
    stamp = ended.strftime("%Y%m%dT%H%M%SZ")
    safe_call_id = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in call_id)
    return f"{TRANSCRIPT_S3_PREFIX}/{date_path}/{safe_call_id}_{stamp}.json"


def upload_transcript_to_s3(transcript: CallTranscript, *, ended_at: float) -> str | None:
    """Upload transcript JSON to S3. Returns s3:// URI or None if bucket unset."""
    if not TRANSCRIPT_S3_BUCKET:
        logger.debug("TRANSCRIPT_S3_BUCKET not set — skipping transcript upload for %s", transcript.call_id)
        return None

    import boto3

    key = transcript_s3_key(transcript.call_id, ended_at)
    region = os.getenv("AWS_DEFAULT_REGION") or os.getenv("AWS_REGION") or "ap-southeast-1"
    client = boto3.client("s3", region_name=region)
    client.put_object(
        Bucket=TRANSCRIPT_S3_BUCKET,
        Key=key,
        Body=transcript.to_json().encode("utf-8"),
        ContentType="application/json",
        Metadata={
            "call-id": transcript.call_id[:1024],
            "agent": transcript.agent[:1024],
        },
    )
    uri = f"s3://{TRANSCRIPT_S3_BUCKET}/{key}"
    logger.info(
        "Uploaded call transcript for %s (%d turns) → %s",
        transcript.call_id,
        len(transcript.turns),
        uri,
    )
    return uri


async def persist_transcript(transcript: CallTranscript, *, ended_at: float) -> str | None:
    """Persist transcript to S3 without blocking the event loop."""
    try:
        return await asyncio.to_thread(upload_transcript_to_s3, transcript, ended_at=ended_at)
    except Exception:
        logger.exception("Failed to upload transcript for call %s", transcript.call_id)
        return None
