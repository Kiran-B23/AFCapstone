import os
from dotenv import load_dotenv

load_dotenv()   # local dev: read GOOGLE_API_KEY from .env (Render injects it as an env var)

import gradio as gr, shutil
from guardrails import redact_pii, make_safe
from agent import agent, retriever

def advise(question, csv_file):
    # Step 2: redact + stage the upload the checker tool will read
    if csv_file:
        with open(csv_file.name) as f: cleaned = redact_pii(f.read())
        with open("data/_current_upload.csv", "w") as f: f.write(cleaned)
    question = redact_pii(question)

    # Steps 3–7: the agent runs (LangSmith records automatically)
    raw = agent.run(question)

    # Step 8: make the answer safe before returning it
    sources = [d.page_content[:120] for d in retriever.invoke(question)]
    return make_safe(raw, sources)

demo = gr.Interface(
    fn=advise,
    inputs=[gr.Textbox(label="Your question"), gr.File(label="Upload transactions (CSV)")],
    outputs=gr.Markdown(label="AdvisorIQ says"),
    title="AdvisorIQ — Wealth-Management Copilot",
    description="Ask an investment question and (optionally) upload your transactions.",
)

if __name__ == "__main__":
    # Render (and most PaaS) inject the port to bind; 0.0.0.0 makes it externally reachable.
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))