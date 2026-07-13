# Voice AI Builder Demo

A local Gradio demo that lets a user create and edit a structured voice-agent configuration through chat, deploy or update the agent in ElevenLabs, and record mock meeting bookings in memory.

## Features

- Chat-based assistant creation and editing
- Live editable JSON configuration
- No database; all state resets when the app restarts
- Create/update an ElevenLabs Conversational AI agent
- Link to the ElevenLabs agent page for browser voice testing
- Local mock meeting-booking flow

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Add your OpenAI and ElevenLabs keys to `.env`. Set `ELEVENLABS_VOICE_ID` to a voice ID from your ElevenLabs account, or leave it blank to use the platform default.

Run:

```bash
python app.py
```

Open the local Gradio URL shown in the terminal.

## Suggested demo

1. Ask: `Build a friendly assistant that qualifies SaaS leads and books demos.`
2. Ask: `Make it more concise and ask about company size before budget.`
3. Review or edit the JSON configuration.
4. Click **Deploy / Update ElevenLabs agent**.
5. Open the generated ElevenLabs URL and test the agent by voice.
6. Add a mock booking in the accordion to show the successful outcome.

## Notes

The ElevenLabs API occasionally changes payload details. Deployment errors are shown directly in the UI, including the API response, making any account-specific adjustment straightforward.
# SalesAI
