import os
from typing import Any

import requests
from dotenv import load_dotenv

load_dotenv()

ELEVENLABS_BASE_URL = "https://api.elevenlabs.io/v1/convai/agents"


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
- Use only the company information provided above.
- Never invent or assume products, services, pricing, policies, availability,
  customers, guarantees, or other company details.
- If a lead asks for information that is not provided, say that you do not
  have that information and offer to arrange a follow-up.
- Ask one question at a time.
- Keep responses concise and conversational.
- Do not claim that a meeting has been booked unless a booking tool confirms it.
""".strip()


def build_elevenlabs_payload(
    config: dict[str, Any],
) -> dict[str, Any]:
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

    headers = {
        "xi-api-key": api_key,
        "Content-Type": "application/json",
    }

    payload = build_elevenlabs_payload(config)

    try:
        if agent_id:
            response = requests.patch(
                f"{ELEVENLABS_BASE_URL}/{agent_id}",
                headers=headers,
                json=payload,
                timeout=30,
            )
            action = "updated"
        else:
            response = requests.post(
                f"{ELEVENLABS_BASE_URL}/create",
                headers=headers,
                json=payload,
                timeout=30,
            )
            action = "created"

        response.raise_for_status()
        response_data = response.json()

        deployed_agent_id = response_data.get(
            "agent_id",
            agent_id,
        )

        return {
            "success": True,
            "agent_id": deployed_agent_id,
            "message": f"Agent {action}: {deployed_agent_id}",
            "test_url": (
                "https://elevenlabs.io/app/"
                f"conversational-ai/{deployed_agent_id}"
            ),
        }

    except requests.RequestException as exc:
        if exc.response is not None:
            detail = exc.response.text
        else:
            detail = str(exc)

        return {
            "success": False,
            "message": f"ElevenLabs deployment failed: {detail}",
        }


def mock_book_meeting(
    name: str,
    email: str,
    requested_time: str,
) -> dict[str, Any]:
    if not name.strip():
        return {
            "success": False,
            "message": "Lead name is required.",
        }

    if not email.strip():
        return {
            "success": False,
            "message": "Lead email is required.",
        }

    if not requested_time.strip():
        return {
            "success": False,
            "message": "Requested time is required.",
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