from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    date: str = Field(
        ...,
        examples=["1991-01-01"],
        description="Target date, YYYY-MM-DD",
    )

    recent_temps: list[float] = Field(
        ...,
        min_length=30,
        description="Recent daily min temps, newest first. Index 0 = yesterday.",
    )


class PredictionResponse(BaseModel):
    date: str
    prediction: float