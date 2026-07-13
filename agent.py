import copy
import json
import os
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from tools import deploy_to_elevenlabs, mock_book_meeting

load_dotenv()

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini")

INITIAL_MESSAGE = """
Before I create your voice agent, please provide:

1. Company name
2. What the company does
3. Who the agent will call
4. The purpose of the calls
5. Information the agent should collect
6. When the agent should offer a meeting
7. Any important facts, restrictions, or wording it must follow

I will not assume or invent any company information.
""".strip()

DEFAULT_CONFIG = {
    "company_name": None,
    "company_description": None,
    "target_leads": None,
    "call_goal": None,
    "information_to_collect": [],
    "meeting_criteria": None,
    "company_facts": [],
    "agent_name": None,
    "voice_id": os.getenv("ELEVENLABS_VOICE_ID", ""),
    "tone": None,
    "opening_message": None,
}

REQUIRED_FIELDS = [
    "company_name",
    "company_description",
    "target_leads",
    "call_goal",
    "information_to_collect",
    "meeting_criteria",
]

BUILDER_PROMPT = """
You are a strict voice-agent configuration assistant.

Your job is to collect information from the user and update a configuration.

Rules:
- Never invent, assume, infer, or guess company information.
- Only add information explicitly stated by the user.
- A company name alone does not tell you what the company does.
- Do not create products, services, customers, pricing, policies, goals,
  qualification questions, or booking rules unless the user explicitly provides them.
- Preserve existing configuration values unless the user explicitly changes them.
- Keep unknown scalar values as null.
- Keep unknown list values as empty lists.
- If required information is missing, ask for it in the reply.
- Do not claim the assistant is ready until every required field is complete.
- The opening message may only use facts present in the configuration.
- Do not place invented examples inside the configuration.

Return only valid JSON with this exact structure:

{
  "config": {
    "company_name": null,
    "company_description": null,
    "target_leads": null,
    "call_goal": null,
    "information_to_collect": [],
    "meeting_criteria": null,
    "company_facts": [],
    "agent_name": null,
    "voice_id": "",
    "tone": null,
    "opening_message": null
  },
  "reply": "Your response to the user"
}
""".strip()


def initial_state() -> dict[str, Any]:
    return {
        "config": copy.deepcopy(DEFAULT_CONFIG),
        "elevenlabs_agent_id": None,
        "bookings": [],
    }


def initial_history() -> list[dict[str, str]]:
    return [
        {
            "role": "assistant",
            "content": INITIAL_MESSAGE,
        }
    ]


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


def missing_required_fields(config: dict[str, Any]) -> list[str]:
    missing = []

    for field in REQUIRED_FIELDS:
        value = config.get(field)

        if value is None:
            missing.append(field)
        elif isinstance(value, str) and not value.strip():
            missing.append(field)
        elif isinstance(value, list) and not value:
            missing.append(field)

    return missing


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    expected_fields = set(DEFAULT_CONFIG)
    received_fields = set(config)

    if received_fields != expected_fields:
        missing = expected_fields - received_fields
        extra = received_fields - expected_fields

        details = []

        if missing:
            details.append(f"missing: {', '.join(sorted(missing))}")

        if extra:
            details.append(f"unexpected: {', '.join(sorted(extra))}")

        raise ValueError("; ".join(details))

    for field in ("information_to_collect", "company_facts"):
        value = config[field]

        if not isinstance(value, list):
            raise ValueError(f"{field} must be a list")

        if not all(isinstance(item, str) for item in value):
            raise ValueError(f"Every value in {field} must be a string")

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
        {
            "role": "user",
            "content": message,
        }
    ]

    try:
        response = get_openai_client().responses.create(
            model=OPENAI_MODEL,
            instructions=BUILDER_PROMPT,
            input=(
                "Current configuration:\n"
                f"{json.dumps(state['config'], indent=2)}\n\n"
                "Conversation:\n"
                f"{json.dumps(history, indent=2)}\n\n"
                "Update the configuration using only facts explicitly "
                "provided in this conversation."
            ),
        )

        result = parse_json(response.output_text)
        updated_config = validate_config(result["config"])
        missing = missing_required_fields(updated_config)

        state = copy.deepcopy(state)
        state["config"] = updated_config

        reply = result["reply"]

        if missing:
            readable_fields = ", ".join(
                field.replace("_", " ")
                for field in missing
            )

            reply += (
                "\n\nI still need the following before the agent can "
                f"be deployed: {readable_fields}."
            )
        else:
            reply += (
                "\n\nAll required information has been provided. "
                "Please review the configuration before deploying."
            )

    except Exception as exc:
        reply = f"I couldn't update the configuration: {exc}"

    history.append(
        {
            "role": "assistant",
            "content": reply,
        }
    )

    return history, state, state["config"], reply, ""


def deploy_agent(state: dict[str, Any]):
    missing = missing_required_fields(state["config"])

    if missing:
        readable_fields = ", ".join(
            field.replace("_", " ")
            for field in missing
        )

        return (
            state,
            f"Cannot deploy yet. Missing: {readable_fields}.",
            "",
        )

    state = copy.deepcopy(state)

    result = deploy_to_elevenlabs(
        config=state["config"],
        agent_id=state.get("elevenlabs_agent_id"),
    )

    if result["success"]:
        state["elevenlabs_agent_id"] = result["agent_id"]

    return state, result["message"], result.get("test_url", "")


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
        initial_history(),
        state,
        state["config"],
        [],
        "Demo reset.",
        "",
        "",
    )