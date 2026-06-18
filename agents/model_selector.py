from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

from agents.schemas import ModelSelectorOutput


load_dotenv()


def run_model_selector(
    user_problem: str,
    planner_output: str,
    feature_engineer_output: str
) -> ModelSelectorOutput:
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.5
    )

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are an expert ML model selection specialist.

            Given a problem description and prior analysis, recommend the top 2-3
            machine learning models.

            For each model provide:
            1. Why it is suitable for this problem
            2. Strengths and limitations
            3. Key hyperparameters to tune

            Also provide:
            - The most appropriate evaluation metric
            - Why that metric should be optimized

            Be specific and justify every recommendation.
            Return output strictly following the provided schema.
            """
        ),
        (
            "human",
            """Problem Description:
            {problem}

            Planner Analysis:
            {planner_output}

            Feature Engineering Output:
            {feature_engineer_output}
            """
        )
    ])

    chain = prompt | llm.with_structured_output(ModelSelectorOutput)

    result = chain.invoke({
        "problem": user_problem,
        "planner_output": planner_output,
        "feature_engineer_output": feature_engineer_output
    })

    return result

if __name__ == "__main__":
    from agents.planner import run_planner
    from agents.feature_engineer import run_feature_engineer
    problem = "Predict house prices based on area, location, and number of bedrooms."
    planner_output = run_planner(problem)
    feature_output = run_feature_engineer(problem, planner_output)
    result = run_model_selector(problem, planner_output, feature_output)
    print(result)
    for model in result.recommended_models:
        print(f"- {model.name}: {model.reasoning}")
    print(f"Metric: {result.evaluation_metric}")