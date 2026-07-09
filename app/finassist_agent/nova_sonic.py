"""AWS Bedrock Nova Sonic bidirectional streaming client."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import struct
import time
import uuid
from typing import TYPE_CHECKING, AsyncIterator, Callable, Optional

if TYPE_CHECKING:
    from finassist_agent.transcript_logger import TranscriptCollector

logger = logging.getLogger(__name__)

# AgentDuet phone leg: 24 kHz. Nova Sonic 2: 16 kHz in, 24 kHz out (aria_bedrock_agent.py).
AGENTDUET_SAMPLE_RATE = 24000
NOVA_INPUT_SAMPLE_RATE = 16000
NOVA_OUTPUT_SAMPLE_RATE = 24000
ENDPOINTING_SENSITIVITY = os.getenv("NOVA_ENDPOINTING_SENSITIVITY", "HIGH")


def downsample_24k_to_16k(pcm_24k: bytes) -> bytes:
    """Resample 24 kHz 16-bit mono PCM to 16 kHz for Nova input."""
    if len(pcm_24k) < 2:
        return pcm_24k
    samples = struct.unpack(f"<{len(pcm_24k) // 2}h", pcm_24k)
    n_out = int(len(samples) * NOVA_INPUT_SAMPLE_RATE / AGENTDUET_SAMPLE_RATE)
    ratio = AGENTDUET_SAMPLE_RATE / NOVA_INPUT_SAMPLE_RATE
    out: list[int] = []
    for i in range(n_out):
        src = i * ratio
        idx = int(src)
        frac = src - idx
        if idx + 1 < len(samples):
            val = int(samples[idx] * (1 - frac) + samples[idx + 1] * frac)
        else:
            val = samples[min(idx, len(samples) - 1)]
        out.append(max(-32768, min(32767, val)))
    return struct.pack(f"<{len(out)}h", *out)

class NovaSonicSession:
    """Manages a single Nova Sonic bidirectional stream session."""

    def __init__(
        self,
        system_prompt: str,
        *,
        model_id: str = "amazon.nova-2-sonic-v1:0",
        region: str = "ap-northeast-1",
        voice_id: str = "matthew",
        transcript_collector: Optional["TranscriptCollector"] = None,
    ):
        self.model_id = model_id
        self.region = region
        self.voice_id = voice_id
        self.system_prompt = system_prompt

        self.prompt_name = str(uuid.uuid4())
        self.content_name = str(uuid.uuid4())
        self.audio_content_name = str(uuid.uuid4())

        self._client = None
        self._stream = None
        self._active = False
        self._closing = False
        self._role = "ASSISTANT"
        self._on_transcript: Optional[Callable[[str, str], None]] = None
        self._transcript_collector = transcript_collector
        self._last_transcript: dict[str, tuple[str, float]] = {}
        self._started_at = 0.0
        self._first_audio_logged = False
        self._event_queue: Optional[asyncio.Queue] = None
        self._pump_task: Optional[asyncio.Task] = None

    def _initialize_client(self) -> None:
        from aws_sdk_bedrock_runtime.client import BedrockRuntimeClient
        from aws_sdk_bedrock_runtime.config import Config
        from smithy_aws_core.identity import EnvironmentCredentialsResolver

        config = Config(
            endpoint_uri=f"https://bedrock-runtime.{self.region}.amazonaws.com",
            region=self.region,
            aws_credentials_identity_resolver=EnvironmentCredentialsResolver(),
        )
        self._client = BedrockRuntimeClient(config=config)

    async def _send_event(self, payload: dict | str) -> None:
        if self._stream is None:
            return
        if not self._active and not self._closing:
            return
        from aws_sdk_bedrock_runtime.models import (
            BidirectionalInputPayloadPart,
            InvokeModelWithBidirectionalStreamInputChunk,
        )

        raw = json.dumps(payload) if isinstance(payload, dict) else payload
        chunk = InvokeModelWithBidirectionalStreamInputChunk(
            value=BidirectionalInputPayloadPart(bytes_=raw.encode("utf-8"))
        )
        try:
            await self._stream.input_stream.send(chunk)
        except Exception:
            if not self._closing:
                raise

    async def _send_session_setup(self) -> None:
        session_start: dict = {
            "event": {
                "sessionStart": {
                    "inferenceConfiguration": {
                        "maxTokens": 512,
                        "topP": 0.9,
                        "temperature": 0.7,
                    },
                }
            }
        }
        if "nova-2-sonic" in self.model_id.lower():
            session_start["event"]["sessionStart"]["turnDetectionConfiguration"] = {
                "endpointingSensitivity": ENDPOINTING_SENSITIVITY,
            }
        await self._send_event(session_start)

        await self._send_event(
            {
                "event": {
                    "promptStart": {
                        "promptName": self.prompt_name,
                        "textOutputConfiguration": {"mediaType": "text/plain"},
                        "audioOutputConfiguration": {
                            "mediaType": "audio/lpcm",
                            "sampleRateHertz": NOVA_OUTPUT_SAMPLE_RATE,
                            "sampleSizeBits": 16,
                            "channelCount": 1,
                            "voiceId": self.voice_id,
                            "encoding": "base64",
                            "audioType": "SPEECH",
                        },
                    }
                }
            }
        )

        await self._send_event(
            {
                "event": {
                    "contentStart": {
                        "promptName": self.prompt_name,
                        "contentName": self.content_name,
                        "type": "TEXT",
                        "interactive": True,
                        "role": "SYSTEM",
                        "textInputConfiguration": {"mediaType": "text/plain"},
                    }
                }
            }
        )
        await self._send_event(
            {
                "event": {
                    "textInput": {
                        "promptName": self.prompt_name,
                        "contentName": self.content_name,
                        "content": self.system_prompt,
                    }
                }
            }
        )
        await self._send_event(
            {
                "event": {
                    "contentEnd": {
                        "promptName": self.prompt_name,
                        "contentName": self.content_name,
                    }
                }
            }
        )

        await self._send_event(
            {
                "event": {
                    "contentStart": {
                        "promptName": self.prompt_name,
                        "contentName": self.audio_content_name,
                        "type": "AUDIO",
                        "interactive": True,
                        "role": "USER",
                        "audioInputConfiguration": {
                            "mediaType": "audio/lpcm",
                            "sampleRateHertz": NOVA_INPUT_SAMPLE_RATE,
                            "sampleSizeBits": 16,
                            "channelCount": 1,
                            "audioType": "SPEECH",
                            "encoding": "base64",
                        },
                    }
                }
            }
        )

    async def _trigger_immediate_greeting(self) -> None:
        """Speak the opening line without waiting for caller endpointing."""
        content_name = str(uuid.uuid4())
        await self._send_event(
            {
                "event": {
                    "contentStart": {
                        "promptName": self.prompt_name,
                        "contentName": content_name,
                        "type": "TEXT",
                        "interactive": True,
                        "role": "USER",
                        "textInputConfiguration": {"mediaType": "text/plain"},
                    }
                }
            }
        )
        await self._send_event(
            {
                "event": {
                    "textInput": {
                        "promptName": self.prompt_name,
                        "contentName": content_name,
                        "content": "The call just connected. Deliver your opening greeting now.",
                    }
                }
            }
        )
        await self._send_event(
            {
                "event": {
                    "contentEnd": {
                        "promptName": self.prompt_name,
                        "contentName": content_name,
                    }
                }
            }
        )

    async def _pump_events(self) -> None:
        try:
            while self._active:
                output = await self._stream.await_output()
                result = await output[1].receive()
                if not result.value or not result.value.bytes_:
                    continue
                data = json.loads(result.value.bytes_.decode("utf-8"))
                event = data.get("event", {})

                if self._transcript_collector is not None and event:
                    self._transcript_collector.process_event(event)

                if "contentStart" in event:
                    self._role = event["contentStart"].get("role", self._role)

                if "textOutput" in event:
                    text = event["textOutput"].get("content", "")
                    role = event["textOutput"].get("role", self._role)
                    if not self._is_interruption_marker(text):
                        self._emit_transcript(role, text)

                if "audioOutput" in event and not self._first_audio_logged and self._started_at:
                    elapsed = time.monotonic() - self._started_at
                    logger.info("First Nova audio to caller after %.2fs", elapsed)
                    self._first_audio_logged = True

                await self._event_queue.put(data)
        except asyncio.CancelledError:
            raise
        except Exception:
            if not self._closing:
                logger.exception("Nova event pump error")
        finally:
            await self._event_queue.put(None)

    async def prepare(self) -> None:
        from aws_sdk_bedrock_runtime.client import InvokeModelWithBidirectionalStreamOperationInput

        if self._client is None:
            self._initialize_client()

        self._started_at = time.monotonic()
        self._event_queue = asyncio.Queue()
        logger.info(
            "Nova Sonic starting model=%s region=%s voice=%s in=%dkHz out=%dkHz endpointing=%s",
            self.model_id,
            self.region,
            self.voice_id,
            NOVA_INPUT_SAMPLE_RATE // 1000,
            NOVA_OUTPUT_SAMPLE_RATE // 1000,
            ENDPOINTING_SENSITIVITY if "nova-2-sonic" in self.model_id.lower() else "n/a",
        )

        self._stream = await self._client.invoke_model_with_bidirectional_stream(
            InvokeModelWithBidirectionalStreamOperationInput(model_id=self.model_id)
        )
        self._active = True
        self._pump_task = asyncio.create_task(self._pump_events())
        await self._send_session_setup()
        await self._trigger_immediate_greeting()
        logger.info("Nova prepared in %.2fs", time.monotonic() - self._started_at)

    async def start(self) -> None:
        await self.prepare()

    async def cancel(self) -> None:
        self._active = False
        self._closing = True
        if self._pump_task and not self._pump_task.done():
            self._pump_task.cancel()
            await asyncio.gather(self._pump_task, return_exceptions=True)
        await self.close()

    async def send_audio(self, pcm_24k: bytes) -> None:
        if not self._active or not pcm_24k or self._closing:
            return
        pcm_16k = downsample_24k_to_16k(pcm_24k)
        encoded = base64.b64encode(pcm_16k).decode("utf-8")
        await self._send_event(
            {
                "event": {
                    "audioInput": {
                        "promptName": self.prompt_name,
                        "contentName": self.audio_content_name,
                        "content": encoded,
                    }
                }
            }
        )

    @staticmethod
    def _is_interruption_marker(text: str) -> bool:
        return '{ "interrupted" : true }' in text or '"interrupted"' in text and "true" in text

    def _emit_transcript(self, role: str, text: str) -> None:
        cleaned = text.strip()
        if not cleaned or not self._on_transcript:
            return
        import time

        now = time.monotonic()
        last_text, last_at = self._last_transcript.get(role, ("", 0.0))
        if cleaned == last_text and (now - last_at) < 2.0:
            return
        self._last_transcript[role] = (cleaned, now)
        self._on_transcript(role, cleaned)

    async def receive(self) -> AsyncIterator[dict]:
        """Yield parsed Nova output events from the background pump."""
        while True:
            data = await self._event_queue.get()
            if data is None:
                break
            yield data

    async def close(self) -> None:
        if self._closing:
            return
        self._closing = True
        self._active = False

        if self._pump_task and not self._pump_task.done():
            self._pump_task.cancel()
            await asyncio.gather(self._pump_task, return_exceptions=True)

        try:
            await self._send_event(
                {
                    "event": {
                        "contentEnd": {
                            "promptName": self.prompt_name,
                            "contentName": self.audio_content_name,
                        }
                    }
                }
            )
            await self._send_event({"event": {"promptEnd": {"promptName": self.prompt_name}}})
            await self._send_event({"event": {"sessionEnd": {}}})
        except Exception:
            logger.debug("Nova sessionEnd events skipped", exc_info=True)
        finally:
            self._active = False
            if self._stream and self._stream.input_stream:
                try:
                    await self._stream.input_stream.close()
                except Exception:
                    logger.debug("Nova input stream close error", exc_info=True)
