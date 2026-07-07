"""AgentCore entrypoint — FinAssist voice agent (AgentDuet + Nova Sonic 2)."""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv

# Secrets for local dev: finassist/agentcore/.env.local
# override=True so finassist/agentcore/.env.local wins over shell / repo-root .env
load_dotenv(
    Path(__file__).resolve().parent.parent / "agentcore" / ".env.local",
    override=True,
)

if sys.version_info < (3, 12):
    sys.exit("Requires Python 3.12+ for AWS Nova Sonic.")

from bedrock_agentcore.runtime import BedrockAgentCoreApp
from bedrock_agentcore.runtime.models import PingStatus

from finassist_agent.logic import evaluate_claim, evaluate_loan, parse_dollar_amount
from finassist_agent.prompts import AGENT_NAME, PLATFORM_NAME
from finassist_agent.voice_service import VoiceAgentService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

voice_service = VoiceAgentService()
_listener_task: asyncio.Task | None = None
_listener_task_id: int | None = None


@asynccontextmanager
async def lifespan(app: BedrockAgentCoreApp):
    """Start AgentDuet voice listener when AgentCore starts."""
    global _listener_task, _listener_task_id

    _listener_task_id = app.add_async_task(
        "agentduet_nova_listener",
        {"connector": os.getenv("AGENTDUET_CONNECTOR_UUID", "")},
    )
    _listener_task = asyncio.create_task(voice_service.run_forever())
    logger.info("%s — AgentDuet + Nova Sonic listener started", AGENT_NAME)
    yield

    voice_service.request_shutdown()
    if _listener_task and not _listener_task.done():
        _listener_task.cancel()
        try:
            await _listener_task
        except asyncio.CancelledError:
            pass
    if _listener_task_id is not None:
        app.complete_async_task(_listener_task_id)
    logger.info("%s listener stopped", AGENT_NAME)


app = BedrockAgentCoreApp(lifespan=lifespan)


@app.ping
def health_check():
    if voice_service.active_call_count > 0 or voice_service.status()["connected"]:
        return PingStatus.HEALTHY_BUSY
    return PingStatus.HEALTHY


@app.entrypoint
def agent_invocation(payload, context):
    action = payload.get("action", "status")

    if action == "status":
        return {
            "agent": AGENT_NAME,
            "platform": PLATFORM_NAME,
            "stack": ["AgentDuet", "AWS Nova Sonic 2", "Bedrock AgentCore"],
            "session_id": context.session_id,
            **voice_service.status(),
            "message": "FinAssist is listening for inbound calls.",
        }

    if action == "evaluate_loan":
        amount = payload.get("amount")
        income = payload.get("annual_income")
        if amount is None or income is None:
            return {"error": "Provide amount and annual_income"}
        if isinstance(amount, str):
            amount = parse_dollar_amount(amount) or float(amount.replace(",", ""))
        if isinstance(income, str):
            income = parse_dollar_amount(income) or float(income.replace(",", ""))
        decision = evaluate_loan(
            amount=float(amount),
            annual_income=float(income),
            purpose=str(payload.get("purpose", "")),
        )
        return {
            "status": decision.status.value,
            "action": decision.action,
            "summary": decision.summary,
        }

    if action == "evaluate_claim":
        policy = payload.get("policy_number")
        description = payload.get("description")
        if not policy or not description:
            return {"error": "Provide policy_number and description"}
        decision = evaluate_claim(
            policy_number=str(policy),
            description=str(description),
            incident_date=str(payload.get("incident_date", "")),
        )
        return {
            "status": decision.status.value,
            "action": decision.action,
            "summary": decision.summary,
        }

    return {
        "error": f"Unknown action: {action}",
        "supported_actions": ["status", "evaluate_loan", "evaluate_claim"],
    }


if __name__ == "__main__":
    app.run()
