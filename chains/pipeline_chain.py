from agents.planner import run_planner
from agents.feature_engineer import run_feature_engineer
from agents.model_selector import run_model_selector
from agents.code_generator import run_code_generator
from agents.code_executor import execute_with_correction


def run_pipeline(user_problem: str) -> dict:
    # ---------------------------
    # Step 1: Planner (always runs)
    # ---------------------------
    planner_output = run_planner(user_problem)

    tools = planner_output.tools_needed if planner_output else []

    # ---------------------------
    # Step 2: Feature Engineering (conditional)
    # ---------------------------
    feature_output = None
    if "feature_engineering" in tools or "model_selection" in tools:
        feature_output = run_feature_engineer(
            user_problem=user_problem,
            planner_output=planner_output
        )

    # ---------------------------
    # Step 3: Model Selection (conditional)
    # ---------------------------
    model_output = None
    if "model_selection" in tools:
        model_output = run_model_selector(
            user_problem=user_problem,
            planner_output=planner_output,
            feature_engineer_output=feature_output
        )

    # ---------------------------
    # Step 4: Code Generation (FIXED — now conditional)
    # ---------------------------
    code_output = None
    if "code_generation" in tools:
        code_output = run_code_generator(
            user_problem=user_problem,
            planner_output=planner_output,
            feature_engineer_output=feature_output,
            model_selector_output=model_output
        )

    # ---------------------------
    # Step 5: Execution (only if code exists)
    # ---------------------------
    execution_result = None
    if code_output:
        execution_result = execute_with_correction(code_output)

    return {
        "planner": planner_output,
        "feature_engineer": feature_output,
        "model_selector": model_output,
        "code_generator": code_output,
        "execution": execution_result
    }


