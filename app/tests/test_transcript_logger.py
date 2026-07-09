"""Tests for Nova Sonic transcript collection."""

from finassist_agent.transcript_logger import (
    CallTranscript,
    TranscriptCollector,
    transcript_s3_key,
)


def _user_turn(content: str, content_id: str = "user-1") -> list:
    return [
        {
            "contentStart": {
                "contentId": content_id,
                "role": "USER",
                "type": "TEXT",
                "textOutputConfiguration": {"mediaType": "text/plain"},
                "additionalModelFields": '{"generationStage":"FINAL"}',
            }
        },
        {"textOutput": {"contentId": content_id, "role": "USER", "content": content}},
        {"contentEnd": {"contentId": content_id, "type": "TEXT", "stopReason": "PARTIAL_TURN"}},
    ]


def _assistant_turn(content: str, content_id: str = "asst-1", *, speculative: bool = False) -> list:
    stage = "SPECULATIVE" if speculative else "FINAL"
    return [
        {
            "contentStart": {
                "contentId": content_id,
                "role": "ASSISTANT",
                "type": "TEXT",
                "textOutputConfiguration": {"mediaType": "text/plain"},
                "additionalModelFields": f'{{"generationStage":"{stage}"}}',
            }
        },
        {"textOutput": {"contentId": content_id, "role": "ASSISTANT", "content": content}},
        {"contentEnd": {"contentId": content_id, "type": "TEXT"}},
    ]


def test_collects_user_and_assistant_final_turns() -> None:
    collector = TranscriptCollector()
    for event in _user_turn("I need a loan") + _assistant_turn("Sure, how much?"):
        collector.process_event(event)

    assert len(collector.turns) == 2
    assert collector.turns[0].role == "USER"
    assert collector.turns[0].content == "I need a loan"
    assert collector.turns[1].role == "ASSISTANT"
    assert collector.turns[1].content == "Sure, how much?"
    assert collector.turns[1].generation_stage == "FINAL"


def test_skips_speculative_assistant_preview() -> None:
    collector = TranscriptCollector()
    for event in _assistant_turn("draft response", speculative=True):
        collector.process_event(event)

    assert collector.turns == []


def test_skips_interruption_marker_and_internal_trigger() -> None:
    collector = TranscriptCollector()
    collector.process_event(
        {"textOutput": {"role": "ASSISTANT", "content": '{ "interrupted" : true }'}}
    )
    collector.process_event(
        {
            "textOutput": {
                "role": "USER",
                "content": "The call just connected. Deliver your opening greeting now.",
            }
        }
    )

    assert collector.turns == []


def test_flush_pending_finalizes_open_turn() -> None:
    collector = TranscriptCollector()
    collector.process_event(_user_turn("hello", "u1")[0])
    collector.process_event(_user_turn("hello", "u1")[1])
    assert collector.turns == []

    collector.flush_pending()
    assert len(collector.turns) == 1
    assert collector.turns[0].content == "hello"


def test_call_transcript_json_shape() -> None:
    collector = TranscriptCollector()
    for event in _user_turn("loan please"):
        collector.process_event(event)

    transcript = CallTranscript.build(
        call_id="call-123",
        caller_number="+15551234567",
        started_at=1_000_000.0,
        ended_at=1_000_045.5,
        collector=collector,
        agent="FinAssist",
        platform="SecureFinance",
        nova_model="amazon.nova-2-sonic-v1:0",
        nova_region="ap-northeast-1",
    )
    payload = transcript.to_dict()

    assert payload["call_id"] == "call-123"
    assert payload["turn_count"] == 1
    assert payload["turns"][0]["role"] == "USER"


def test_transcript_s3_key_layout() -> None:
    key = transcript_s3_key("call/abc", 1_735_689_600.0)
    assert key.startswith("finassist/transcripts/")
    assert key.endswith(".json")
    assert "call_abc" in key
