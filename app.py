import json
import os
from copy import deepcopy
from typing import Any

import gradio as gr
import requests
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

DEFAULT_CONFIG = {
    "name": "Sales Qualifier",
    "tone": "Friendly and professional",
    "goal": "Qualify leads and book a meeting when there is a good fit.",
    "first_message": "Hi, this is Alex. Is now a good time for a quick conversation?",
    "voice_id": os.getenv("ELEVENLABS_VOICE_ID", ""),
    "qualification_questions": [
        "What problem are you trying to solve?",
        "What is your timeline?",
        "What budget range have you set aside?",
    ],
    "booking_instructions": "If the lead is interested and qualified, collect their name, email, and preferred meeting time, then confirm the booking.",
}

SYSTEM_PROMPT = """You configure a voice AI assistant from natural-language instructions.
Return only valid JSON with exactly these keys:
name, tone, goal, first_message, voice_id, qualification_questions, booking_instructions.
qualification_questions must be an array of short strings.
Preserve useful existing values unless the user asks to change them.
Keep the assistant safe, concise, and suitable for a demo.
"""


def initial_state() -> dict[str, Any]:
    return {"config": deepcopy(DEFAULT_CONFIG), "agent_id": None, "bookings": []}


def build_agent_prompt(config: dict[str, Any]) -> str:
    questions = "\n".join(
        f"{index + 1}. {question}"
        for index, question in enumerate(config["qualification_questions"])
    )
    return f"""You are {config['name']}.

Goal:
{config['goal']}

Tone:
{config['tone']}

Qualification questions:
{questions}

Booking behavior:
{config['booking_instructions']}

Conversation rules:
- Be natural and concise.
- Ask one question at a time.
- Do not invent facts.
- Politely end the conversation if the person is not interested.
- For this demo, say that a meeting has been booked only after collecting a name, email, and preferred time.
"""


def normalize_config(raw: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(current)
    for key in result:
        if key in raw and raw[key] is not None:
            result[key] = raw[key]

    if not isinstance(result["qualification_questions"], list):
        result["qualification_questions"] = current["qualification_questions"]
    result["qualification_questions"] = [
        str(question).strip()
        for question in result["qualification_questions"]
        if str(question).strip()
    ]
    for key in ("name", "tone", "goal", "first_message", "voice_id", "booking_instructions"):
        result[key] = str(result[key]).strip()
    return result


def update_builder(message: str, history: list[list[str | None]], state: dict[str, Any]):
    if not message.strip():
        return history, state, state["config"], "Please enter an instruction.", ""

    history = history + [[message, None]]
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        reply = "OpenAI is not configured. Add OPENAI_API_KEY to your .env file."
        history[-1][1] = reply
        return history, state, state["config"], reply, ""

    try:
        client = OpenAI(api_key=api_key)
        response = client.responses.create(
            model=os.getenv("OPENAI_MODEL", "gpt-5-mini"),
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "current_config": state["config"],
                            "requested_change": message,
                        }
                    ),
                },
            ],
            text={"format": {"type": "json_object"}},
        )
        updated = normalize_config(json.loads(response.output_text), state["config"])
        state = deepcopy(state)
        state["config"] = updated
        reply = f"Updated **{updated['name']}**. You can refine it further or deploy it to ElevenLabs."
    except Exception as exc:
        reply = f"I couldn't update the configuration: {exc}"

    history[-1][1] = reply
    return history, state, state["config"], reply, ""


def reset_demo():
    state = initial_state()
    return [], state, state["config"], "Demo reset.", ""


def elevenlabs_payload(config: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "name": config["name"],
        "conversation_config": {
            "agent": {
                "first_message": config["first_message"],
                "language": "en",
                "prompt": {"prompt": build_agent_prompt(config)},
            }
        },
    }
    if config["voice_id"]:
        payload["conversation_config"]["tts"] = {"voice_id": config["voice_id"]}
    return payload


def deploy_agent(state: dict[str, Any]):
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        return state, "ElevenLabs is not configured. Add ELEVENLABS_API_KEY to your .env file.", gr.update(visible=False)

    headers = {"xi-api-key": api_key, "Content-Type": "application/json"}
    payload = elevenlabs_payload(state["config"])
    agent_id = state.get("agent_id")

    try:
        if agent_id:
            response = requests.patch(
                f"https://api.elevenlabs.io/v1/convai/agents/{agent_id}",
                headers=headers,
                json=payload,
                timeout=30,
            )
            action = "Updated"
        else:
            response = requests.post(
                "https://api.elevenlabs.io/v1/convai/agents/create",
                headers=headers,
                json=payload,
                timeout=30,
            )
            action = "Created"

        response.raise_for_status()
        data = response.json()
        state = deepcopy(state)
        state["agent_id"] = data.get("agent_id", agent_id)
        url = f"https://elevenlabs.io/app/conversational-ai/{state['agent_id']}"
        status = f"{action} ElevenLabs agent `{state['agent_id']}`. Open it to run a browser voice test."
        return state, status, gr.update(value=url, visible=True)
    except requests.HTTPError:
        detail = response.text[:1200]
        return state, f"ElevenLabs rejected the request ({response.status_code}): {detail}", gr.update(visible=False)
    except Exception as exc:
        return state, f"Deployment failed: {exc}", gr.update(visible=False)


def save_manual_config(config: dict[str, Any], state: dict[str, Any]):
    try:
        updated = normalize_config(config, state["config"])
        state = deepcopy(state)
        state["config"] = updated
        return state, updated, "Manual changes saved locally."
    except Exception as exc:
        return state, state["config"], f"Could not save changes: {exc}"


def mock_booking(name: str, email: str, requested_time: str, state: dict[str, Any]):
    if not all(value.strip() for value in (name, email, requested_time)):
        return state, state["bookings"], "Fill in all booking fields."

    booking = {"name": name.strip(), "email": email.strip(), "requested_time": requested_time.strip()}
    state = deepcopy(state)
    state["bookings"].append(booking)
    return state, state["bookings"], f"Mock meeting booked for {booking['requested_time']}."


CSS = """
.gradio-container {max-width: 1250px !important;}
#hero {text-align: center; margin-bottom: 10px;}
#status {min-height: 44px;}
"""

with gr.Blocks(title="Voice AI Builder") as demo:
    state = gr.State(initial_state())

    gr.Markdown(
        "# Voice AI Builder\nDescribe an assistant, refine it through chat, deploy it to ElevenLabs, and test a local mock booking flow.",
        elem_id="hero",
    )

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(height=520, label="Builder chat")
            message = gr.Textbox(
                placeholder="Build a friendly assistant that qualifies real-estate buyers...",
                label="Instruction",
            )
            with gr.Row():
                send = gr.Button("Send", variant="primary")
                reset = gr.Button("Reset")

        with gr.Column(scale=2):
            config = gr.JSON(value=DEFAULT_CONFIG, label="Live assistant configuration")
            save = gr.Button("Save manual edits")
            deploy = gr.Button("Deploy / Update ElevenLabs agent", variant="primary")
            agent_link = gr.Textbox(label="ElevenLabs agent URL", visible=False, interactive=False)
            status = gr.Markdown("Ready.", elem_id="status")

    with gr.Accordion("Mock meeting booking", open=False):
        gr.Markdown("This keeps the demo local and proves the booking outcome without a database or calendar integration.")
        with gr.Row():
            booking_name = gr.Textbox(label="Lead name")
            booking_email = gr.Textbox(label="Lead email")
            booking_time = gr.Textbox(label="Preferred time")
        book = gr.Button("Book mock meeting")
        bookings = gr.JSON(value=[], label="Bookings in this session")

    send.click(
        update_builder,
        inputs=[message, chatbot, state],
        outputs=[chatbot, state, config, status, message],
    )
    message.submit(
        update_builder,
        inputs=[message, chatbot, state],
        outputs=[chatbot, state, config, status, message],
    )
    reset.click(reset_demo, outputs=[chatbot, state, config, status, agent_link])
    save.click(save_manual_config, inputs=[config, state], outputs=[state, config, status])
    deploy.click(deploy_agent, inputs=[state], outputs=[state, status, agent_link])
    book.click(
        mock_booking,
        inputs=[booking_name, booking_email, booking_time, state],
        outputs=[state, bookings, status],
    )

if __name__ == "__main__":
    demo.launch(css=CSS)
