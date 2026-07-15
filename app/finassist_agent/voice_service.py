"""AgentDuet + Nova Sonic voice bridge for FinAssist."""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

from agentduet import (
    Call,
    CallAudioConfig,
    InboundCallMode,
    IncomingCallNotification,
    SessionManager,
    SessionManagerConfig,
    TriggerConditionsBuilder,
    new_session_id,
)
from agentduet.exceptions import BufferFullError, CallNotFoundError

from finassist_agent.nova_config import resolve_nova_settings
from finassist_agent.nova_sonic import NovaSonicSession
from finassist_agent.prompts import (
    AGENT_NAME,
    PLATFORM_NAME,
    build_call_system_prompt,
    is_hangup_request,
)
from finassist_agent.transcript_logger import CallTranscript, TranscriptCollector, persist_transcript

logger = logging.getLogger(__name__)

AGENTDUET_API_KEY = os.getenv("AGENTDUET_API_KEY")
AGENTDUET_CONNECTOR_UUID = os.getenv("AGENTDUET_CONNECTOR_UUID")
NOVA_VOICE_ID = os.getenv("NOVA_SONIC_VOICE_ID", "matthew")
HANGUP_GRACE_SECONDS = float(os.getenv("HANGUP_GRACE_SECONDS", "2.5"))


def _nova_settings() -> tuple[str, str]:
    """Read Nova config at runtime (after load_dotenv)."""
    return resolve_nova_settings()


def require_python_312() -> None:
    if sys.version_info < (3, 12):
        raise RuntimeError(
            "Nova Sonic requires Python 3.12+ (aws_sdk_bedrock_runtime). "
            "Use: python3.12 -m venv .venv312"
        )


def _caller_number(call: Call) -> str:
    """External party number/handle for logs and transcripts."""
    return str(call.caller) or call.participant.value


async def attach_inbound_call(sm: SessionManager, noti: IncomingCallNotification) -> Call:
    """Open an ephemeral session and attach it to the notified inbound call.

    agentduet 1.0.0b9 pattern: ``open_session`` + ``process_call`` (no client registry).
    """
    session = await sm.open_session(new_session_id(), noti.subscriber)
    try:
        return await session.process_call(noti)
    except CallNotFoundError:
        # Rare race: notification expired/taken before attach — one fresh retry.
        session = await sm.open_session(new_session_id(), noti.subscriber)
        return await session.process_call(noti)


async def bridge_call_to_nova(call: Call) -> None:
    """Full-duplex bridge: AgentDuet 24 kHz ↔ Nova Sonic (16 kHz in / 24 kHz out)."""
    nova_region, nova_model = _nova_settings()
    call_started_at = time.time()
    transcript_collector = TranscriptCollector()
    caller_number = _caller_number(call)

    nova = NovaSonicSession(
        build_call_system_prompt(),
        model_id=nova_model,
        region=nova_region,
        voice_id=NOVA_VOICE_ID,
        transcript_collector=transcript_collector,
    )
    nova._initialize_client()  # noqa: SLF001

    stop = asyncio.Event()
    hangup_requested = asyncio.Event()
    hangup_task: Optional[asyncio.Task] = None
    last_user_at: float = 0.0

    async def schedule_hangup() -> None:
        await asyncio.sleep(HANGUP_GRACE_SECONDS)
        logger.info("Hang-up grace elapsed — closing call %s", call.id)
        stop.set()

    def on_transcript(role: str, text: str) -> None:
        nonlocal hangup_task, last_user_at
        now = time.monotonic()
        if role == "USER":
            last_user_at = now
            logger.info("[%s] %s", role, text)
        elif role == "ASSISTANT":
            if last_user_at:
                logger.info("[%s] (%.2fs after user) %s", role, now - last_user_at, text)
            else:
                logger.info("[%s] %s", role, text)
        else:
            logger.info("[%s] %s", role, text)
        if role == "USER" and is_hangup_request(text) and not hangup_requested.is_set():
            hangup_requested.set()
            logger.info("Caller requested hang-up")
            if hangup_task is None or hangup_task.done():
                hangup_task = asyncio.create_task(schedule_hangup())

    nova._on_transcript = on_transcript  # noqa: SLF001

    @call.on_hangup
    def on_hangup(_payload: object = None) -> None:
        logger.info("Call %s terminated", call.id)
        stop.set()

    # Warm Nova while the phone leg is answering.
    answer_task = asyncio.create_task(call.answer())
    prepare_task = asyncio.create_task(nova.prepare())

    if not await answer_task:
        logger.error("Failed to answer call %s", call.id)
        await nova.cancel()
        if not prepare_task.done():
            prepare_task.cancel()
            await asyncio.gather(prepare_task, return_exceptions=True)
        return

    try:
        await prepare_task
    except Exception:
        logger.exception("Failed to start Nova Sonic for call %s", call.id)
        await nova.close()
        return

    logger.info(
        "Call %s — %s from %s | model=%s region=%s voice=%s",
        call.id,
        AGENT_NAME,
        caller_number,
        nova_model,
        nova_region,
        NOVA_VOICE_ID,
    )

    async def stream_to_nova() -> None:
        """Full duplex: always forward caller mic → Nova (Nova handles barge-in)."""
        try:
            # Inbound: remote party is call.caller (isolated track 0 in b9).
            async for chunk in call.caller.audio_stream():
                if stop.is_set():
                    break
                await nova.send_audio(chunk)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Error sending caller audio to Nova Sonic")
        finally:
            stop.set()

    async def receive_from_nova() -> None:
        """Stream Nova audio directly to caller — no intermediate queue."""
        try:
            async for data in nova.receive():
                if stop.is_set():
                    break
                ev = data.get("event", {})

                if "textOutput" in ev:
                    text = ev["textOutput"].get("content", "")
                    if nova._is_interruption_marker(text):
                        await call.clear_send_audio_buffer()
                        logger.debug("Nova barge-in — audio buffer cleared")

                if "audioOutput" in ev:
                    audio_bytes = base64.b64decode(ev["audioOutput"]["content"])
                    try:
                        await call.send_audio(audio_bytes)
                    except BufferFullError:
                        logger.warning("AgentDuet send buffer full — dropping audio chunk")
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Error playing Nova Sonic audio to caller")
        finally:
            stop.set()

    try:
        await asyncio.gather(stream_to_nova(), receive_from_nova())
    finally:
        stop.set()
        await nova.close()
        call_ended_at = time.time()
        transcript_collector.flush_pending()
        transcript = CallTranscript.build(
            call_id=str(call.id),
            caller_number=caller_number,
            started_at=call_started_at,
            ended_at=call_ended_at,
            collector=transcript_collector,
            agent=AGENT_NAME,
            platform=PLATFORM_NAME,
            nova_model=nova_model,
            nova_region=nova_region,
        )
        await persist_transcript(transcript, ended_at=call_ended_at)
        try:
            await call.close()
        except Exception:
            logger.debug("Call %s already closed", call.id, exc_info=True)


@dataclass
class VoiceAgentService:
    """Long-running AgentDuet listener bridged to Nova Sonic for FinAssist."""

    _sm: Optional[SessionManager] = None
    _sm_id: Optional[str] = None
    _connected: bool = False
    _active_calls: int = 0
    _inflight: set[str] = field(default_factory=set)
    _shutdown: asyncio.Event = field(default_factory=asyncio.Event)

    @property
    def active_call_count(self) -> int:
        return self._active_calls

    def status(self) -> dict:
        nova_region, nova_model = _nova_settings()
        return {
            "agent": AGENT_NAME,
            "platform": PLATFORM_NAME,
            "connected": self._connected,
            "session_manager_id": self._sm_id,
            "active_calls": self._active_calls,
            "capabilities": ["loan_application", "insurance_claim"],
            "nova_model": nova_model,
            "nova_region": nova_region,
            "nova_voice": NOVA_VOICE_ID,
        }

    async def handle_incoming_call(
        self, sm: SessionManager, noti: IncomingCallNotification
    ) -> None:
        call = await attach_inbound_call(sm, noti)
        self._active_calls += 1
        logger.info("Incoming call %s from %s", call.id, _caller_number(call))
        try:
            await bridge_call_to_nova(call)
        except Exception:
            logger.exception("Unhandled error on call %s", call.id)
        finally:
            self._active_calls -= 1

    async def run_forever(self) -> None:
        require_python_312()

        if not AGENTDUET_API_KEY or not AGENTDUET_CONNECTOR_UUID:
            raise RuntimeError("Set AGENTDUET_API_KEY and AGENTDUET_CONNECTOR_UUID")

        config = SessionManagerConfig.create(
            api_key=AGENTDUET_API_KEY,
            connector_uuid=AGENTDUET_CONNECTOR_UUID,
            call_audio=CallAudioConfig(sample_rate=24000, buffer_size=1024 * 1024),
        )

        # install_signal_handlers=False: AgentCore owns process signals / lifespan.
        async with SessionManager(config) as sm:
            self._sm = sm
            self._connected = True
            self._sm_id = sm.id
            nova_region, nova_model = _nova_settings()
            logger.info(
                "%s (%s) connected | model=%s region=%s voice=%s | SM=%s",
                AGENT_NAME,
                PLATFORM_NAME,
                nova_model,
                nova_region,
                NOVA_VOICE_ID,
                sm.id,
            )

            try:
                await sm.setup_trigger_conditions(
                    TriggerConditionsBuilder().inbound_call(InboundCallMode.ALL).build()
                )
            except Exception as exc:
                logger.warning("Trigger setup failed (%s); continuing", exc)

            @sm.on_incoming_call
            async def on_call(noti: IncomingCallNotification) -> None:
                # SM already runs this handler in a detached task; dedupe by call id.
                if noti.call_id in self._inflight:
                    return
                self._inflight.add(noti.call_id)
                try:
                    await self.handle_incoming_call(sm, noti)
                finally:
                    self._inflight.discard(noti.call_id)

            await sm.run_forever(install_signal_handlers=False)

        self._sm = None
        self._connected = False
        self._sm_id = None

    def request_shutdown(self) -> None:
        """Wake run_forever via SessionManager.disconnect (AgentCore lifespan)."""
        self._shutdown.set()
        sm = self._sm
        if sm is not None:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                return
            loop.create_task(sm.disconnect())
