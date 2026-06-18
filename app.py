import gradio as gr
import os
import tempfile
from dotenv import load_dotenv
from chains.pipeline_chain import run_pipeline
from notes.fetcher import extract_from_pdf, extract_from_url, search_topic
from notes.notes_chain import generate_notes
from rag.indexer import build_vectorstore
from rag.qa_chain import answer_question
from utils.formatters import format_planner, format_feature_engineer, format_model_selector

load_dotenv()


def analyze_pipeline(user_problem):
    if not user_problem.strip():
        return "Please enter a problem description.", "", "", "", ""
    try:
        results = run_pipeline(user_problem)
        planner_str = format_planner(results["planner"]) if results["planner"] else "Not required."
        feature_str = format_feature_engineer(results["feature_engineer"]) if results["feature_engineer"] else "Not required."
        model_str = format_model_selector(results["model_selector"]) if results["model_selector"] else "Not required."
        code_str = results["code_generator"] if results["code_generator"] else "Not required."
        code_str = code_str.replace("```python", "").replace("```", "").strip()
        exec_result = results["execution"]
        if exec_result:
            if exec_result["success"]:
                exec_str = f"Code executed successfully in {exec_result['attempts']} attempt(s)."
            else:
                exec_str = f"Execution failed after {exec_result['attempts']} attempts.\nLast error: {exec_result['error']}"
        else:
            exec_str = "Code generation was not required for this problem."
        return planner_str, feature_str, model_str, code_str, exec_str
    except Exception as e:
        return f"Error: {str(e)}", "", "", "", ""


def build_notes(topic, pdf_file, url, use_web_search, output_format):
    if not topic.strip():
        return "Please enter a topic.", None
    try:
        combined_content = ""
        if pdf_file is not None:
            combined_content += extract_from_pdf(pdf_file.name) + "\n"
        if url and url.strip():
            combined_content += extract_from_url(url) + "\n"
        if use_web_search:
            combined_content += search_topic(topic) + "\n"
        fmt = "latex" if output_format == "LaTeX" else "plain"
        notes = generate_notes(topic, combined_content, fmt)
        if fmt == "latex":
            tmp = tempfile.NamedTemporaryFile(
                delete=False, suffix=".tex", mode="w", encoding="utf-8"
            )
            tmp.write(notes)
            tmp.close()
            return notes, tmp.name
        return notes, None
    except Exception as e:
        return f"Error: {str(e)}", None


pdf_memory = {}  # stores chat_history list per PDF

def answer_pdf_question(pdf_file, question):
    if pdf_file is None:
        return "Please upload a PDF."
    if not question.strip():
        return "Please enter a question."
    try:
        pdf_key = pdf_file.name
        if pdf_key not in pdf_memory:
            pdf_memory[pdf_key] = []
        vectorstore = build_vectorstore(pdf_file.name)
        answer = answer_question(vectorstore, question, pdf_memory[pdf_key])
        return answer
    except Exception as e:
        return f"Error: {str(e)}"


with gr.Blocks(title="ML Pipeline Architect", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# ML Pipeline Architect")
    gr.Markdown("An autonomous ML assistant powered by LangChain + Groq")

    with gr.Tabs():

        # ── Tab 1 ──────────────────────────────────────────────
        with gr.Tab("ML Pipeline"):
            gr.Markdown("### Describe your ML problem and let the agent build a complete pipeline.")
            with gr.Row():
                problem_input = gr.Textbox(
                    label="Describe your ML Problem",
                    lines=5,
                    placeholder="e.g. I want to predict customer churn using transaction history..."
                )
            analyze_btn = gr.Button("Analyze", variant="primary")
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Problem Analysis")
                    planner_out = gr.Markdown()
                with gr.Column():
                    gr.Markdown("### Feature Engineering")
                    feature_out = gr.Markdown()
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Model Selection")
                    model_out = gr.Markdown()
                with gr.Column():
                    gr.Markdown("### Generated Code")
                    code_out = gr.Textbox(lines=15)
            exec_out = gr.Textbox(label="Execution Status", lines=3)
            analyze_btn.click(
                fn=analyze_pipeline,
                inputs=[problem_input],
                outputs=[planner_out, feature_out, model_out, code_out, exec_out]
            )

        # ── Tab 2 ──────────────────────────────────────────────
        with gr.Tab("Notes Builder"):
            gr.Markdown("### Generate structured notes from any topic, PDF, URL, or web search.")
            topic_input = gr.Textbox(label="Topic", placeholder="e.g. Gradient Boosting")
            with gr.Row():
                pdf_input_notes = gr.File(label="Upload PDF (optional)", file_types=[".pdf"])
                url_input = gr.Textbox(label="Blog/Article URL (optional)")
            with gr.Row():
                web_search_cb = gr.Checkbox(label="Search Web for additional context")
                format_radio = gr.Radio(
                    ["Plain", "LaTeX"],
                    label="Output Format",
                    value="Plain"
                )
            notes_btn = gr.Button("Generate Notes", variant="primary")
            notes_out = gr.Markdown(latex_delimiters=[
                {"left": "$$", "right": "$$", "display": True},
                {"left": "$", "right": "$", "display": False}
            ])
            file_out = gr.File(label="Download .tex file")

            notes_btn.click(
                fn=build_notes,
                inputs=[topic_input, pdf_input_notes, url_input, web_search_cb, format_radio],
                outputs=[notes_out, file_out]
            )

        # ── Tab 3 ──────────────────────────────────────────────
        with gr.Tab("PDF Q&A"):
            gr.Markdown("### Upload a PDF and ask anything about it.")
            pdf_input_rag = gr.File(label="Upload PDF", file_types=[".pdf"])
            question_input = gr.Textbox(label="Your Question", placeholder="e.g. What is the main conclusion?")
            ask_btn = gr.Button("Ask", variant="primary")
            answer_out = gr.Markdown(latex_delimiters=[
                {"left": "$$", "right": "$$", "display": True},
                {"left": "$", "right": "$", "display": False}
            ])

            ask_btn.click(
                fn=answer_pdf_question,
                inputs=[pdf_input_rag, question_input],
                outputs=[answer_out]
            )

if __name__ == "__main__":
    demo.launch()