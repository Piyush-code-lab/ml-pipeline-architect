from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

from agents.schemas import FeatureEngineerOutput


load_dotenv()


def run_feature_engineer(user_problem: str, planner_output: str) -> FeatureEngineerOutput:
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.5
    )

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are an expert ML feature engineer.

            Given a problem description and initial analysis, suggest specific
            feature engineering steps with clear reasoning for each step.

            Be practical and specific.
            Avoid generic advice.
            Focus on features that are likely to improve model performance.

            Return your response using the provided schema.
            """
        ),
        (
            "human",
            """Problem Description:
            {problem}

            Initial Analysis:
            {planner_output}
            """
        )
    ])

    chain = prompt | llm.with_structured_output(
        FeatureEngineerOutput
    )

    result = chain.invoke({
        "problem": user_problem,
        "planner_output": planner_output
    })

    return result
