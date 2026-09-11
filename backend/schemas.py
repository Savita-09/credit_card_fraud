from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt, StrictStr


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    features: dict[str, StrictFloat | StrictInt | StrictStr | None] = Field(min_length=1, max_length=200)
    threshold: float | None = Field(None, ge=0, le=1, allow_inf_nan=False)
    explain: bool = True


class Signal(BaseModel):
    feature: str
    value: float | str | None
    reference: float | str | None
    score_delta: float


class PredictionResponse(BaseModel):
    id: str
    prediction: Literal["Fraud", "Legitimate"]
    fraud_probability: float = Field(ge=0, le=1)
    risk_level: Literal["Low", "Medium", "High"]
    threshold: float
    model_name: str
    model_version: str
    recommendation: str
    explanation_method: str
    signals: list[Signal]
    amount: float | None = None
    created_at: str
