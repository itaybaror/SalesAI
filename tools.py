import os
from typing import Any

import requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "https://api.elevenlabs.io/v1/convai/agents"


# Builds the system prompt used by the deployed ElevenLabs agent.
def build_prompt(config: dict[str, Any]) -> str:
    questions = "\n".join(
        f"{i + 1}. {question}"
        for i, question in enumerate(config["qualification_questions"])
    )

    instructions = "\n".join(
        f"- {instruction}"
        for instruction in config["additional_instructions"]
    ) or "- No additional instructions."

    return f"""
You are {config["agent_name"]}, calling on behalf of {config["company_name"]}.

Goal:
{config["call_goal"]}

Tone:
{config["tone"]}

Qualification questions:
{questions}

A lead is qualified when:
{config["success_criteria"]}

Additional instructions:
{instructions}

Rules:
- Use only the information provided here.
- Never invent company information.
- Ask one question at a time.
- Keep responses concise and conversational.
- Only qualify a lead when the success criteria are met.
""".strip()


# Creates a new ElevenLabs agent or updates an existing one.
def deploy_to_elevenlabs(
    config: dict[str, Any],
    agent_id: str | None,
) -> dict[str, Any]:
    api_key = os.getenv("ELEVENLABS_API_KEY")

    if not api_key:
        return {
            "success": False,
            "message": "ELEVENLABS_API_KEY is missing.",
        }

    payload = {
        "name": config["agent_name"],
        "conversation_config": {
            "agent": {
                "first_message": config["opening_message"],
                "prompt": {"prompt": build_prompt(config)},
            }
        },
    }

    # Only include a voice if one was configured.
    if config.get("voice_id"):
        payload["conversation_config"]["tts"] = {
            "voice_id": config["voice_id"]
        }

    # Creating a new agent or updating existing one
    url = (
        f"{BASE_URL}/{agent_id}"
        if agent_id
        else f"{BASE_URL}/create"
    )

    try:
        response = requests.request(
            method="PATCH" if agent_id else "POST",
            url=url,
            headers={
                "xi-api-key": api_key,
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=30,
        )
        response.raise_for_status()

        deployed_id = response.json().get("agent_id", agent_id)

        return {
            "success": True,
            "agent_id": deployed_id,
            "message": "Agent deployed successfully.",
            "dashboard_url": (
                "https://elevenlabs.io/app/conversational-ai/"
                f"{deployed_id}"
            ),
        }

    # Return a readable error instead of raising an exception.
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