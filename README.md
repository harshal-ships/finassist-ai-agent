# FinAssist

**FinAssist** is a demo voice AI agent for **SecureFinance** — a fictional financial services company. Callers can apply for a **loan** or file an **insurance claim** over the phone. The agent collects details through natural conversation, applies eligibility rules, and gives a clear next step (pre-approval, review, fast-track claim, etc.).

Built with **Amazon Bedrock AgentCore**, **AgentDuet** (telephony), and **Amazon Nova Sonic 2** (speech-to-speech). No separate STT/TTS pipeline — Nova handles listening, reasoning, and speaking in one bidirectional stream.

## What FinAssist does

| Scenario | Collects | Example outcomes |
|---|---|---|
| **Loan** | Amount, purpose, annual income | Pre-Approved, Needs Review, Standard Processing |
| **Insurance claim** | Policy number, incident date, description | High Priority, Premium Member fast-track, Standard Claim |

The agent opens with:

> "Hello! I'm FinAssist from SecureFinance. Are you looking to apply for a loan, or file an insurance claim today?"

Conversation rules are enforced in the prompt (one question per turn, no SSN/password/OTP collection). Deterministic eligibility logic lives in `finassist/app/finassist_agent/logic.py` and can also be invoked over HTTP for testing.

## Repository layout

Source code lives in **`finassist/`** (separate git repo for GitHub):

```
finassist/
├── agentcore/
│   ├── agentcore.json           # Runtime config (Python 3.12, env vars)
│   ├── aws-targets.json.example # Copy → aws-targets.json (your account ID)
│   └── .env.local.example       # Copy → .env.local (AgentDuet + Nova settings)
└── app/
    ├── main.py                  # AgentCore entrypoint + voice listener lifespan
    ├── pyproject.toml
    └── finassist_agent/
        ├── voice_service.py     # AgentDuet ↔ Nova audio bridge
        ├── nova_sonic.py        # Bedrock bidirectional streaming client
        ├── nova_config.py       # Nova region routing
        ├── prompts.py           # System prompt + hang-up detection
        └── logic.py             # Loan / claim eligibility rules
```

## Prerequisites

- Node.js 20+ and `@aws/agentcore` CLI
- Python 3.12+
- AWS credentials with Bedrock access (SSO recommended)
- [AgentDuet](https://pypi.org/project/agentduet/) API key + connector UUID

```bash
npm install -g @aws/agentcore
cd finassist/agentcore/cdk && npm install && cd ../..
cd finassist/app && uv sync --python 3.12 && cd ../..
```

## Configure

1. **AWS target** — copy `finassist/agentcore/aws-targets.json.example` → `aws-targets.json` and set your 12-digit account ID
2. **Secrets** — copy `finassist/agentcore/.env.local.example` → `.env.local` and fill in AgentDuet credentials
3. **AWS auth** — use SSO in your shell (do not commit access keys):

```bash
export AWS_PROFILE=your-sso-profile
aws sso login --profile your-sso-profile
```

## Run locally

```bash
cd finassist
agentcore dev
```

Call your AgentDuet connector number — FinAssist answers over voice.

```bash
agentcore dev '{"action": "status"}'
agentcore dev '{"action": "evaluate_loan", "amount": 15000, "annual_income": 60000, "purpose": "home repair"}'
```

## Deploy to AWS

Requires an IAM role with CloudFormation + AgentCore deploy permissions (Bedrock-only access is not enough).

```bash
cd finassist
agentcore deploy
agentcore invoke '{"action": "status"}'
agentcore logs
```

For cloud runtime, add `AGENTDUET_API_KEY` and `AGENTDUET_CONNECTOR_UUID` to `agentcore.json` env vars or AgentCore credentials — `.env.local` is not packaged in the deploy zip.

## Architecture

| Component | Role |
|---|---|
| **AgentCore Runtime** | Hosts the agent, health checks, lifecycle |
| **AgentDuet** | Inbound phone calls, PCM audio streaming, barge-in |
| **Nova Sonic 2** | Speech-to-speech AI via Bedrock bidirectional API |
| **finassist_agent/** | Voice bridge, prompts, region config, eligibility logic |

