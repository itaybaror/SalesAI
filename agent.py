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
    call_goal: str | None = None
    qualification_questions: list[str] = Field(default_factory=list)
    success_criteria: str | None = None
    opening_message: str | None = None
    additional_instructions: list[str] = Field(default_factory=list)
    tone: str = "Friendly and professional"
    agent_name: str = "Lead Qualification Assistant"
    voice_id: str = ""


class BuilderResponse(BaseModel):
    config: AgentConfig
    reply: str


REQUIRED_FIELDS = {
    "company_name",
    "call_goal",
    "qualification_questions",
    "success_criteria",
    "opening_message",
}


INITIAL_MESSAGE = """
Tell me about the lead qualification agent you need.

Please include:
1. The company name
2. The purpose of the calls
3. The questions the agent should ask
4. What makes a lead qualified
5. What the agent should say when the call begins

You can also provide tone or additional restrictions. I will not invent
company information or qualification rules.
""".strip()


SYSTEM_PROMPT = """
You configure outbound lead qualification voice agents.

Strict rules:
- Use only information explicitly provided by the user.
- Never invent or assume company facts, products, pricing, policies,
  qualification rules, or customer details.
- Preserve existing values unless the user explicitly changes them.
- Leave unknown optional text fields as null.
- Leave unknown lists empty.
- Never generate or change voice_id.
- Ask clearly for any missing required information.
- Do not say the configuration is ready unless all required fields are present.
""".strip()


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


def get_client() -> OpenAI:
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
        response = get_client().responses.parse(
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

        missing = missing_fields(state["config"])
        reply = result.reply

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

    dashboard_link = (
        f"[Open this agent in ElevenLabs]({result['dashboard_url']})"
    )

    return (
        state,
        result["message"],
        result["agent_id"],
        dashboard_link,
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