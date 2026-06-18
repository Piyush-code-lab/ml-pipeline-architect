import pdfplumber
import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS


def extract_from_pdf(pdf_path: str) -> str:
    if not pdf_path:
        return ""

    text_parts = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()

            if page_text:
                text_parts.append(page_text)

    return "\n".join(text_parts)


def extract_from_url(url: str) -> str:
    if not url:
        return ""

    response = requests.get(url, timeout=10)

    soup = BeautifulSoup(
        response.content,
        "html.parser"
    )

    return soup.get_text()


def search_topic(topic: str) -> str:
    if not topic:
        return ""

    results = DDGS().text(
        topic,
        max_results=5
    )

    snippets = []

    for result in results:
        snippets.append(
            result.get("body", "")
        )

    return "\n".join(snippets)