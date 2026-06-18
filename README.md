---
title: ML Pipeline Architect
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 5.33.0
app_file: app.py
pinned: false
---

# ML Pipeline Architect

An autonomous multi-agent ML assistant powered by LangChain + Groq.

## Features

- **Tab 1 - ML Pipeline**: Describe any ML problem and get a complete pipeline with feature engineering, model selection, and executable code — with self-correction
- **Tab 2 - Notes Builder**: Generate structured notes from any topic, PDF, URL, or web search — with LaTeX math rendering
- **Tab 3 - PDF Q&A**: Upload any PDF and ask questions with conversation memory

## Tech Stack

- LangChain + Groq (Llama 3.3 70B)
- Pydantic structured outputs
- ChromaDB + HuggingFace embeddings
- Gradio