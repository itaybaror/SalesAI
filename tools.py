import os
from typing import Any

import requests
from dotenv import load_dotenv

load_dotenv()

ELEVENLABS_BASE_URL = "https://api.elevenlabs.io/v1/convai/agents"


def build_agent_prompt(config: dict[str, Any]) -> str:
    questions = "\n".join(
        f"{index + 1}. {question}"
        for index, question in enumerate(
            config["qualification_questions"]
        )
    )

    return f"""
You are {config["name"]}, a voice-based sales qualification assistant.

Goal:
{config["goal"]}

Tone:
{config["tone"]}

Qualification questions:
{questions}

Booking instructions:
{config["booking_instructions"]}

Ask one question at a time.
Keep responses concise and conversational.
Do not claim that a meeting is booked unless a booking tool confirms it.
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
            "voice_id": config["voice_id"]
        }

    return {
        "name": config["name"],
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
            "message": (
                f"Agent {action}: {deployed_agent_id}"
            ),
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
            "message": (
                f"ElevenLabs deployment failed: {detail}"
            ),
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
        "message": (
            f"Mock meeting booked for "
            f"{requested_time.strip()}."
        ),
    }