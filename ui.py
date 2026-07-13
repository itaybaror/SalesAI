import gradio as gr

from agent import (
    book_meeting,
    deploy_agent,
    initial_state,
    reset_demo,
    update_builder,
)

CSS = """
.gradio-container {
    max-width: 1400px !important;
}

#status {
    min-height: 44px;
}
"""


with gr.Blocks(title="Voice AI Builder") as demo:
    state = gr.State(initial_state())

    gr.Markdown(
        "# Voice AI Builder\n"
        "Describe the sales assistant you want, refine it through "
        "chat, and deploy it to ElevenLabs."
    )

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                height=520,
                label="Builder chat",
            )

            message = gr.Textbox(
                label="Instruction",
                placeholder=(
                    "Build a friendly assistant that qualifies "
                    "real-estate buyers..."
                ),
            )

            send_button = gr.Button(
                "Send",
                variant="primary",
            )

        with gr.Column(scale=2):
            config_preview = gr.JSON(
                value=initial_state()["config"],
                label="Current assistant configuration",
            )

            deploy_button = gr.Button(
                "Deploy to ElevenLabs",
                variant="primary",
            )

            test_url = gr.Textbox(
                label="ElevenLabs test URL",
                interactive=False,
            )

    status = gr.Markdown(elem_id="status")

    with gr.Accordion(
        "Mock meeting booking",
        open=False,
    ):
        with gr.Row():
            lead_name = gr.Textbox(
                label="Lead name"
            )

            lead_email = gr.Textbox(
                label="Lead email"
            )

            requested_time = gr.Textbox(
                label="Requested time"
            )

        booking_button = gr.Button(
            "Book mock meeting"
        )

        bookings = gr.JSON(
            value=[],
            label="Local bookings",
        )

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
            test_url,
        ],
    )

    booking_button.click(
        fn=book_meeting,
        inputs=[
            lead_name,
            lead_email,
            requested_time,
            state,
        ],
        outputs=[
            state,
            bookings,
            status,
        ],
    )

    reset_button.click(
        fn=reset_demo,
        outputs=[
            chatbot,
            state,
            config_preview,
            bookings,
            status,
            test_url,
            message,
        ],
    )