from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

from agents.schemas import PlannerOutput


load_dotenv()


def run_planner(user_problem: str) -> PlannerOutput:
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.5
    )

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are an expert ML problem analyst.

            Given a problem description, identify:
            1. Problem type (classification, regression, clustering, recommendation, forecasting, etc.)
            2. Target variable (if mentioned)
            3. Key challenges
            4. Recommended preprocessing and feature engineering steps
            5. Tools needed from:
            - feature_engineering
            - model_selection
            - code_generation

            Be concise, structured, and practical.
            """
        ),
        (
            "human",
            "Problem Description:\n{problem}"
        )
    ])

    chain = prompt | llm.with_structured_output(PlannerOutput)

    result = chain.invoke({
        "problem": user_problem
    })

    return result

