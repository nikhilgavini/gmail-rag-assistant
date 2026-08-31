import gradio as gr
from dotenv import load_dotenv

from gmail_assistant.answer import answer_question

load_dotenv(override=True)

## Formats the context pane
def format_context(context):
    if not context:
        return "*No relevant context found*"

    result = "<h2 style='color: #ff7800;'>Relevant Context</h2>\n\n"

    for doc in context:
        result += f"<span style='color: #ff7800;'>Source: {doc.metadata['source']}</span>\n\n"
        result += doc.page_content + "\n\n"
    return result


def chat(history):
    last_message = history[-1]["content"]  # pass raw, let answer_question clean it
    prior = history[:-1]
    history.append({"role": "assistant", "content": ""})
    
    token_generator = answer_question(last_message, prior)
    
    for accumulated_answer, context in token_generator:
        if accumulated_answer is None:
            accumulated_answer = "Sorry, I couldn't answer that question as an error occurred."
            context = []

        # Update the last message in history with the newly grown text
        history[-1]["content"] = accumulated_answer
        
        # Format the context window. It will pop up as soon as Chroma answers (instant), 
        # while the text underneath continues to stream in.
        formatted_ctx = format_context(context)

        # Yield BOTH components to update the UI on every single token
        yield history, formatted_ctx


def main():
    def put_message_in_chatbot(message, history):
        if isinstance(message, dict):
            text_content = message.get("text", "")
        else:
            text_content = str(message)
        return "", history + [{"role": "user", "content": text_content}]

    # Change theme to whatever user wants
    theme = gr.themes.Ocean(font=["Inter", "system-ui", "sans-serif"])

    # Title
    with gr.Blocks() as ui:
        gr.Markdown("# Gmail Expert Assistant\nAsk me anything about your emails!")
        gr.Markdown("### Created by Nikhil Gavini\nEmail: nikhilgavini@gmail.com")

        with gr.Row():
            with gr.Column(scale=1):
                chatbot = gr.Chatbot(
                    label="Conversation", 
                    height=600,
                    resizable=True,
                    buttons=["copy", "copy_all"]
                )
                message = gr.Textbox(
                    label="Your Question",
                    placeholder="Ask anything about your emails...",
                    show_label=False,
                )

            with gr.Column(scale=1):
                context_markdown = gr.Markdown(
                    label="Retrieved Context",
                    value="*Retrieved context will appear here*",
                    container=True,
                    height=600,
                    buttons=["copy", "copy_all"]
                )

        message.submit(
            put_message_in_chatbot, inputs=[message, chatbot], outputs=[message, chatbot]
        ).then(chat, inputs=chatbot, outputs=[chatbot, context_markdown])

    ui.launch(
        theme=theme,
        server_name = "0.0.0.0",
        server_port = 7860,
        inbrowser = False, 
        share = False
    )

if __name__ == "__main__":
    main()