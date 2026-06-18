from pydantic import BaseModel


class PlannerOutput(BaseModel):
    problem_type: str
    target_variable: str
    key_challenges: list[str]
    preprocessing_steps: list[str]
    tools_needed: list[str]


class FeatureSchema(BaseModel):
    name: str
    description: str
    reasoning: str


class FeatureEngineerOutput(BaseModel):
    features: list[FeatureSchema]


class ModelSchema(BaseModel):
    name: str
    reasoning: str
    hyperparameters: list[str]


class ModelSelectorOutput(BaseModel):
    recommended_models: list[ModelSchema]
    evaluation_metric: str
    metric_reasoning: str