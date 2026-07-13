import os
from typing import Any

import requests
from dotenv import load_dotenv

load_dotenv()

ELEVENLABS_AGENTS_URL = "https://api.elevenlabs.io/v1/convai/agents"


def build_agent_prompt(config: dict[str, Any]) -> str:
    questions = "\n".join(
        f"{index + 1}. {question}"
        for index, question in enumerate(
            config["qualification_questions"]
        )
    )

    instructions = "\n".join(
        f"- {instruction}"
        for instruction in config["additional_instructions"]
    )

    if not instructions:
        instructions = "- No additional instructions were provided."

    return f"""
You are {config["agent_name"]}, calling on behalf of
{config["company_name"]}.

Call goal:
{config["call_goal"]}

Tone:
{config["tone"]}

Qualification questions:
{questions}

A lead is successful or qualified when:
{config["success_criteria"]}

Additional instructions:
{instructions}

Rules:
- Use only the information provided in this prompt.
- Never invent company details, pricing, policies, products, or guarantees.
- If asked for unavailable information, say you do not have it.
- Ask one qualification question at a time.
- Keep responses concise and conversational.
- Do not say a lead is qualified unless the success criteria are met.
- Do not say a meeting is booked unless a booking tool confirms it.
""".strip()


def build_elevenlabs_payload(
    config: dict[str, Any],
) -> dict[str, Any]:
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

        if not deployed_agent_id:
            raise ValueError("ElevenLabs returned no agent ID.")

        return {
            "success": True,
            "agent_id": deployed_agent_id,
            "message": f"Agent {action} successfully.",
            "dashboard_url": (
                "https://elevenlabs.io/app/conversational-ai/"
                f"{deployed_agent_id}"
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