from agents.schemas import (
    PlannerOutput,
    FeatureEngineerOutput,
    ModelSelectorOutput
)

def format_planner(output: PlannerOutput) -> str:
    challenges = "\n".join(f"- {c}" for c in output.key_challenges)
    steps = "\n".join(f"- {s}" for s in output.preprocessing_steps)
    tools = "\n".join(f"- {t}" for t in output.tools_needed)
    return f"### Problem Analysis\n- Problem Type: {output.problem_type}\n- Target Variable: {output.target_variable}\n\n### Key Challenges\n{challenges}\n\n### Preprocessing Steps\n{steps}\n\n### Tools Selected\n{tools}"

def format_feature_engineer(output: FeatureEngineerOutput) -> str:
    lines = ["### Recommended Features"]
    for f in output.features:
        lines.append(f"#### {f.name}\n- Description: {f.description}\n- Reasoning: {f.reasoning}")
    return "\n\n".join(lines)

def format_model_selector(output: ModelSelectorOutput) -> str:
    lines = ["### Recommended Models"]
    for m in output.recommended_models:
        hyperparams = ", ".join(m.hyperparameters)
        lines.append(f"#### {m.name}\n- Reasoning: {m.reasoning}\n- Hyperparameters: {hyperparams}")
    lines.append(f"### Evaluation Metric\n- Metric: {output.evaluation_metric}\n- Reasoning: {output.metric_reasoning}")
    return "\n\n".join(lines)