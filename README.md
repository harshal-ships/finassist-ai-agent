# AgentCore + AgentDuet Voice Agent

Build a phone-call AI agent with **Amazon Bedrock AgentCore**, **AgentDuet** (telephony), and **Nova Sonic 2** (speech-to-speech).

```
Phone → AgentDuet → AgentCore Runtime → Nova Sonic 2 → FinAssist
```

## Project structure

```
finassist/
├── agentcore/
│   ├── agentcore.json       # runtime config
│   ├── aws-targets.json     # AWS account + region
│   └── .env.local           # secrets (gitignored)
└── app/
    ├── main.py              # AgentCore entrypoint
    ├── pyproject.toml       # Python dependencies
    └── finassist_agent/     # voice bridge + prompts + Nova client
```

## Prerequisites

- Node.js 20+ and `@aws/agentcore` CLI
- Python 3.12+
- AWS credentials with Bedrock access
- AgentDuet API key + connector UUID

```bash
npm install -g @aws/agentcore
cd finassist/agentcore/cdk && npm install && cd ../..
cd finassist/app && uv sync --python 3.12 && cd ../..
```

## Configure

1. **AWS target** — edit `finassist/agentcore/aws-targets.json` with your 12-digit account ID
2. **Secrets** — copy `finassist/agentcore/.env.local.example` → `finassist/agentcore/.env.local`

## Run locally

```bash
cd finassist
agentcore dev
```

Call your AgentDuet connector number — FinAssist answers over voice.

```bash
agentcore dev '{"action": "status"}'
```

## Deploy to AWS

```bash
cd finassist
agentcore deploy
agentcore invoke '{"action": "status"}'
agentcore logs
```

## Architecture

| Component | Role |
|---|---|
| **AgentCore Runtime** | Hosts the agent, health checks, lifecycle |
| **AgentDuet** | Inbound phone calls, audio streaming |
| **Nova Sonic 2** | Speech-to-speech AI (Bedrock) |
| **finassist_agent/** | Prompts, eligibility logic, voice bridge |

