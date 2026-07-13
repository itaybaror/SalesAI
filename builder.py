import copy
import json
import os
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

from tools import deploy_to_elevenlabs

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

You can also provide a tone or additional restrictions.
I will not invent company information or qualification rules.
""".strip()

SYSTEM_PROMPT = """
You configure outbound lead qualification voice agents.

Rules:
- Use only information explicitly provided by the user.
- Never invent company facts, products, pricing, policies, customers,
  qualification rules, or opening messages.
- Preserve existing values unless the user explicitly changes them.
- Leave unknown text fields as null and unknown lists empty.
- Never generate or modify voice_id.
- Ask clearly for missing required information.
- Do not say the configuration is ready unless every required field exists.
""".strip()


# Creates a fresh local application state.
def initial_state() -> dict[str, Any]:
    config = AgentConfig(
        voice_id=os.getenv("ELEVENLABS_VOICE_ID", "")
    )

    return {
        "config": config.model_dump(),
        "elevenlabs_agent_id": None,
    }


# Returns the builder's initial chat message.
def initial_history() -> list[dict[str, str]]:
    return [
        {
            "role": "assistant",
            "content": INITIAL_MESSAGE,
        }
    ]


# Creates an authenticated OpenAI client.
def get_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing from .env")

    return OpenAI(api_key=api_key)


# Finds required configuration fields that are still empty.
def missing_fields(config: dict[str, Any]) -> list[str]:
    return sorted(
        field
        for field in REQUIRED_FIELDS
        if not config.get(field)
    )


# Updates the configuration from the latest chat message.
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
            fields = ", ".join(
                field.replace("_", " ")
                for field in missing
            )
            reply += f"\n\nStill needed: {fields}."
        else:
            reply += "\n\nThe configuration is ready for review."

    except Exception as exc:
        reply = f"I couldn't update the configuration: {exc}"

    history.append(
        {"role": "assistant", "content": reply}
    )

    return history, state, state["config"], reply, ""


# Creates or updates the configured ElevenLabs agent.
def deploy_agent(state: dict[str, Any]):
    missing = missing_fields(state["config"])

    if missing:
        fields = ", ".join(
            field.replace("_", " ")
            for field in missing
        )

        return state, f"Cannot deploy. Missing: {fields}.", "", ""

    state = copy.deepcopy(state)

    result = deploy_to_elevenlabs(
        config=state["config"],
        agent_id=state.get("elevenlabs_agent_id"),
    )

    if not result["success"]:
        return state, result["message"], "", ""

    state["elevenlabs_agent_id"] = result["agent_id"]

    link = (
        f"[Open this agent in ElevenLabs]"
        f"({result['dashboard_url']})"
    )

    return state, result["message"], result["agent_id"], link


# Resets all local demo state.
def reset_demo():
    state = initial_state()

    return (
        initial_history(),
        state,
        state["config"],
        "Demo reset.",
        "",
        "",
        "",
    )