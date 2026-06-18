from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


load_dotenv()


def generate_notes(
    topic: str,
    content: str,
    output_format: str
) -> str:
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.8
    )

    if output_format.lower() == "latex":
        system_prompt = """
        You are an expert educator and LaTeX writer.

        Generate comprehensive notes on the given topic as clean,
        compilable LaTeX code.

        Include:
        - Proper documentclass
        - Required packages
        - Title
        - Sections and subsections
        - Lists, equations, and examples where appropriate
        - Professional formatting

        If additional context is provided, use it to enrich the notes.
        If not, use your own knowledge.
        For mathematical expressions, use $...$ for inline math and $$...$$ for display math. Never use backticks for math.
        Return only the LaTeX code and nothing else.
        """
    else:
        system_prompt = """
        You are an expert educator.

        Generate comprehensive, well-structured notes on the given topic.

        Requirements:
        - Clear headings
        - Logical organization
        - Bullet points where appropriate
        - Examples when useful
        - Concise explanations of important concepts
        For mathematical expressions, use $...$ for inline math and $$...$$ for display math. Never use backticks for math.

        If additional context is provided, use it to enrich the notes.
        If not, use your own knowledge.
        """

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            system_prompt
        ),
        (
            "human",
            """Topic: {topic}

            Additional Context (use if relevant, ignore if empty):
            {content}

            Generate detailed notes now.
            """
        )
    ])

    parser = StrOutputParser()

    chain = prompt | llm | parser

    result = chain.invoke({
        "topic": topic,
        "content": content
    })

    return result


    