import os
from typing import Any

import requests
from dotenv import load_dotenv

load_dotenv()

ELEVENLABS_AGENTS_URL = "https://api.elevenlabs.io/v1/convai/agents"


def build_agent_prompt(config: dict[str, Any]) -> str:
    information_to_collect = "\n".join(
        f"{index + 1}. {item}"
        for index, item in enumerate(config["information_to_collect"])
    )

    company_facts = "\n".join(
        f"- {fact}"
        for fact in config["company_facts"]
    )

    if not company_facts:
        company_facts = "- No additional company facts were provided."

    return f"""
You are {config["agent_name"]}, a voice-based sales assistant for
{config["company_name"]}.

Company description:
{config["company_description"]}

Target leads:
{config["target_leads"]}

Call goal:
{config["call_goal"]}

Tone:
{config["tone"]}

Information to collect:
{information_to_collect}

Meeting criteria:
{config["meeting_criteria"]}

Approved company facts:
{company_facts}

Rules:
- Use only the information explicitly provided above.
- Never invent company details, products, pricing, policies, or guarantees.
- If information is unavailable, say you do not have that information.
- Ask one question at a time.
- Keep responses concise and conversational.
- Do not claim a meeting was booked unless a tool confirms it.
""".strip()


def build_elevenlabs_payload(
    config: dict[str, Any],
) -> dict[str, Any]:
    if not config.get("agent_name"):
        raise ValueError("Agent name is required.")

    if not config.get("opening_message"):
        raise ValueError("Opening message is required.")

    conversation_config: dict[str, Any] = {
        "agent": {
            "first_message": config["opening_message"],
            "prompt": {
                "prompt": build_agent_prompt(config),
            },
        }
    }

    if config.get("voice_id"):
        conversation_config["tts"] = {
            "voice_id": config["voice_id"],
        }

    return {
        "name": config["agent_name"],
        "conversation_config": conversation_config,
    }


def deploy_to_elevenlabs(
    config: dict[str, Any],
    agent_id: str | None,
) -> dict[str, Any]:
    api_key = os.getenv("ELEVENLABS_API_KEY")

    if not api_key:
        return {
            "success": False,
            "message": "ELEVENLABS_API_KEY is missing from .env",
        }

    try:
        payload = build_elevenlabs_payload(config)

        headers = {
            "xi-api-key": api_key,
            "Content-Type": "application/json",
        }

        if agent_id:
            response = requests.patch(
                f"{ELEVENLABS_AGENTS_URL}/{agent_id}",
                headers=headers,
                json=payload,
                timeout=30,
            )
            action = "updated"
        else:
            response = requests.post(
                f"{ELEVENLABS_AGENTS_URL}/create",
                headers=headers,
                json=payload,
                timeout=30,
            )
            action = "created"

        response.raise_for_status()
        data = response.json()
        deployed_agent_id = data.get("agent_id", agent_id)

        return {
            "success": True,
            "agent_id": deployed_agent_id,
            "message": f"Agent {action} successfully.",
            "dashboard_url": (
                "https://elevenlabs.io/app/conversational-ai"
            ),
        }

    except ValueError as exc:
        return {
            "success": False,
            "message": str(exc),
        }

    except requests.RequestException as exc:
        detail = (
            exc.response.text
            if exc.response is not None
            else str(exc)
        )

        return {
            "success": False,
            "message": f"ElevenLabs deployment failed: {detail}",
        }

def mock_book_meeting(
    name: str,
    email: str,
    requested_time: str,
) -> dict[str, Any]:
    if not name.strip() or not email.strip() or not requested_time.strip():
        return {
            "success": False,
            "message": "Name, email, and requested time are required.",
        }

    booking = {
        "name": name.strip(),
        "email": email.strip(),
        "requested_time": requested_time.strip(),
        "status": "mock-booked",
    }

    return {
        "success": True,
        "booking": booking,
        "message": f"Mock meeting booked for {requested_time.strip()}.",
    }