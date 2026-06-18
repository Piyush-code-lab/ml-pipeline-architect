from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from agents.schemas import PlannerOutput, FeatureEngineerOutput, ModelSelectorOutput


load_dotenv()


def run_code_generator(
    user_problem: str,
    planner_output: PlannerOutput,
    feature_engineer_output: FeatureEngineerOutput,
    model_selector_output: ModelSelectorOutput
) -> str:
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.5
    )

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are an expert ML engineer who writes clean, runnable Python code.

            Generate a complete sklearn pipeline including:
            - Data loading placeholder
            - Preprocessing
            - Feature engineering
            - Model training
            - Evaluation
            - Proper comments

            Return only executable Python code.
            """
        ),
        (
            "human",
            """Problem:
            {problem}

            Planner Output:
            {planner_output}

            Feature Engineering Output:
            {feature_engineer_output}

            Model Selection Output:
            {model_selector_output}
            """
        )
    ])

    chain = prompt | llm | StrOutputParser()

    result = chain.invoke({
        "problem": user_problem,
        "planner_output": str(planner_output.model_dump()) if planner_output else "",
        "feature_engineer_output": str(feature_engineer_output.model_dump()) if feature_engineer_output else "",
        "model_selector_output": str(model_selector_output.model_dump()) if model_selector_output else ""
    })

    return result