import copy
import json
import os
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

from tools import deploy_to_elevenlabs, mock_book_meeting

load_dotenv()

MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini")


class AgentConfig(BaseModel):
    company_name: str | None = None
    company_description: str | None = None
    target_leads: str | None = None
    call_goal: str | None = None
    information_to_collect: list[str] = Field(default_factory=list)
    meeting_criteria: str | None = None
    company_facts: list[str] = Field(default_factory=list)
    agent_name: str | None = None
    voice_id: str = ""
    tone: str | None = None
    opening_message: str | None = None


class BuilderResponse(BaseModel):
    config: AgentConfig
    reply: str


INITIAL_MESSAGE = """
Before I create your voice agent, please tell me:

1. Your company name
2. What the company does
3. Who the agent will speak with
4. The goal of the calls
5. What information it should collect
6. When it should offer a meeting
7. Any important facts or restrictions

I will not assume or invent company information.
""".strip()


SYSTEM_PROMPT = """
You configure voice agents.

Strict rules:
- Use only information explicitly provided by the user.
- Never guess or invent company facts.
- Preserve existing values unless the user changes them.
- Leave unknown text fields as null.
- Leave unknown lists empty.
- Ask for any important missing information.
- Do not say the agent is ready unless the required information is present.
""".strip()


REQUIRED_FIELDS = {
    "company_name",
    "company_description",
    "target_leads",
    "call_goal",
    "information_to_collect",
    "meeting_criteria",
    "agent_name",
    "tone",
    "opening_message",
}


def initial_state() -> dict[str, Any]:
    config = AgentConfig(
        voice_id=os.getenv("ELEVENLABS_VOICE_ID", "")
    )

    return {
        "config": config.model_dump(),
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


def client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing from .env")

    return OpenAI(api_key=api_key)


def missing_fields(config: dict[str, Any]) -> list[str]:
    return [
        field
        for field in REQUIRED_FIELDS
        if not config.get(field)
    ]


def update_builder(
    message: str,
    history: list[dict[str, str]],
    state: dict[str, Any],
):
    if not message.strip():
        return history, state, state["config"], "Enter a message.", ""

    history = history + [
        {"role": "user", "content": message}
    ]

    try:
        response = client().responses.parse(
            model=MODEL,
            instructions=SYSTEM_PROMPT,
            input=(
                "Current configuration:\n"
                f"{json.dumps(state['config'], indent=2)}\n\n"
                "Conversation:\n"
                f"{json.dumps(history, indent=2)}"
            ),
            text_format=BuilderResponse,
        )

        result = response.output_parsed

        if result is None:
            raise RuntimeError("The model returned no structured response.")

        state = copy.deepcopy(state)
        state["config"] = result.config.model_dump()

        reply = result.reply
        missing = missing_fields(state["config"])

        if missing:
            readable = ", ".join(
                field.replace("_", " ")
                for field in missing
            )
            reply += f"\n\nStill needed: {readable}."
        else:
            reply += "\n\nThe configuration is ready for review."

    except Exception as exc:
        reply = f"I couldn't update the configuration: {exc}"

    history.append(
        {"role": "assistant", "content": reply}
    )

    return history, state, state["config"], reply, ""


def deploy_agent(state: dict[str, Any]):
    missing = missing_fields(state["config"])

    if missing:
        readable = ", ".join(
            field.replace("_", " ")
            for field in missing
        )

        return state, f"Cannot deploy. Missing: {readable}.", "", ""

    state = copy.deepcopy(state)

    result = deploy_to_elevenlabs(
        config=state["config"],
        agent_id=state.get("elevenlabs_agent_id"),
    )

    if not result["success"]:
        return state, result["message"], "", ""

    state["elevenlabs_agent_id"] = result["agent_id"]

    return (
        state,
        result["message"],
        result["agent_id"],
        result["dashboard_url"],
    )


def book_meeting(
    name: str,
    email: str,
    requested_time: str,
    state: dict[str, Any],
):
    result = mock_book_meeting(name, email, requested_time)

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
        "",
    )