import gradio as gr

from builder import (
    deploy_agent,
    initial_history,
    initial_state,
    reset_demo,
    update_builder,
)

HEAD = """
<script
    src="https://unpkg.com/@elevenlabs/convai-widget-embed"
    async
    type="text/javascript">
</script>
"""


# Builds the Gradio interface.
def build_ui() -> gr.Blocks:
    state_value = initial_state()

    with gr.Blocks(
        title="Voice AI Builder",
        head=HEAD,
        fill_width=False,
    ) as demo:
        state = gr.State(state_value)

        gr.Markdown(
            "# Voice AI Builder\n"
            "Create and deploy a lead qualification voice agent."
        )

        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    value=initial_history(),
                    height=520,
                    label="Builder chat",
                )

                message = gr.Textbox(
                    label="Instruction",
                    placeholder="Describe the agent your company needs...",
                )

                send_button = gr.Button(
                    "Send",
                    variant="primary",
                )

            with gr.Column(scale=2):
                config_preview = gr.JSON(
                    value=state_value["config"],
                    label="Current assistant configuration",
                )

                deploy_button = gr.Button(
                    "Deploy to ElevenLabs",
                    variant="primary",
                )

                agent_id = gr.Textbox(
                    label="Deployed ElevenLabs agent ID",
                    interactive=False,
                )

                voice_widget = gr.HTML()

        status = gr.Markdown()
        reset_button = gr.Button("Reset demo")

        builder_inputs = [
            message,
            chatbot,
            state,
        ]

        builder_outputs = [
            chatbot,
            state,
            config_preview,
            status,
            message,
        ]

        message.submit(
            fn=update_builder,
            inputs=builder_inputs,
            outputs=builder_outputs,
        )

        send_button.click(
            fn=update_builder,
            inputs=builder_inputs,
            outputs=builder_outputs,
        )

        deploy_button.click(
            fn=deploy_agent,
            inputs=state,
            outputs=[
                state,
                status,
                agent_id,
                voice_widget,
            ],
        )

        reset_button.click(
            fn=reset_demo,
            outputs=[
                chatbot,
                state,
                config_preview,
                status,
                agent_id,
                voice_widget,
                message,
            ],
        )

    return demo


demo = build_ui()