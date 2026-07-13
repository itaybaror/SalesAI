import copy
import json
import os
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from tools import deploy_to_elevenlabs, mock_book_meeting

load_dotenv()

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini")

DEFAULT_CONFIG = {
    "name": "Sales Qualification Assistant",
    "voice_id": os.getenv("ELEVENLABS_VOICE_ID", ""),
    "tone": "friendly and professional",
    "goal": "Qualify leads and book meetings with suitable prospects.",
    "opening_message": (
        "Hi, thanks for taking the call. Is now a good time to speak?"
    ),
    "qualification_questions": [
        "What problem are you trying to solve?",
        "What is your approximate budget?",
        "What is your timeline?",
        "Are you the decision maker?",
    ],
    "booking_instructions": (
        "Offer a meeting when the lead is qualified and interested."
    ),
}

BUILDER_PROMPT = """
You configure sales voice assistants.

Update the current configuration according to the user's latest request.
Preserve values the user did not ask to change.

Return only valid JSON with exactly these fields:
name, voice_id, tone, goal, opening_message,
qualification_questions, booking_instructions.

qualification_questions must be an array of strings.
Do not include Markdown or explanations.
""".strip()


def initial_state() -> dict[str, Any]:
    return {
        "config": copy.deepcopy(DEFAULT_CONFIG),
        "elevenlabs_agent_id": None,
        "bookings": [],
    }


def get_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing from .env")

    return OpenAI(api_key=api_key)


def parse_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()

    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0]

    return json.loads(cleaned)


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    required_fields = set(DEFAULT_CONFIG)
    missing_fields = required_fields - set(config)

    if missing_fields:
        missing = ", ".join(sorted(missing_fields))
        raise ValueError(f"Missing configuration fields: {missing}")

    questions = config["qualification_questions"]

    if not isinstance(questions, list):
        raise ValueError("qualification_questions must be a list")

    if not all(isinstance(question, str) for question in questions):
        raise ValueError("Every qualification question must be a string")

    return {
        field: config[field]
        for field in DEFAULT_CONFIG
    }


def update_builder(
    message: str,
    history: list[dict[str, str]],
    state: dict[str, Any],
):
    if not message.strip():
        return (
            history,
            state,
            state["config"],
            "Please enter an instruction.",
            "",
        )

    history = history + [
        {"role": "user", "content": message}
    ]

    try:
        response = get_openai_client().responses.create(
            model=OPENAI_MODEL,
            instructions=BUILDER_PROMPT,
            input=(
                "Current configuration:\n"
                f"{json.dumps(state['config'], indent=2)}"
                "\n\nUser request:\n"
                f"{message}"
            ),
        )

        updated_config = validate_config(
            parse_json(response.output_text)
        )

        state = copy.deepcopy(state)
        state["config"] = updated_config

        reply = (
            f"Updated {updated_config['name']}. "
            "You can keep editing it or deploy it."
        )
    except Exception as exc:
        reply = f"I couldn't update the configuration: {exc}"

    history.append({
        "role": "assistant",
        "content": reply,
    })

    return history, state, state["config"], reply, ""


def deploy_agent(state: dict[str, Any]):
    state = copy.deepcopy(state)

    result = deploy_to_elevenlabs(
        config=state["config"],
        agent_id=state.get("elevenlabs_agent_id"),
    )

    if result["success"]:
        state["elevenlabs_agent_id"] = result["agent_id"]

    return (
        state,
        result["message"],
        result.get("test_url", ""),
    )


def book_meeting(
    name: str,
    email: str,
    requested_time: str,
    state: dict[str, Any],
):
    result = mock_book_meeting(
        name=name,
        email=email,
        requested_time=requested_time,
    )

    if not result["success"]:
        return state, state["bookings"], result["message"]

    state = copy.deepcopy(state)
    state["bookings"].append(result["booking"])

    return state, state["bookings"], result["message"]


def reset_demo():
    state = initial_state()

    return (
        [],
        state,
        state["config"],
        [],
        "Demo reset.",
        "",
        "",
    )