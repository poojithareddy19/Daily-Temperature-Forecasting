import re
from datetime import date as Date
from typing import Annotated

from pydantic import BaseModel, BeforeValidator, Field

from src.data.validate import TEMP_MAX, TEMP_MIN

ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")


def _iso_date_string(value):
    # Pydantic's own date parsing also takes Unix timestamps and midnight datetimes.
    if not isinstance(value, str) or not ISO_DATE.fullmatch(value):
        raise ValueError("date must be a YYYY-MM-DD string")
    return value


Temperature = Annotated[float, Field(ge=TEMP_MIN, le=TEMP_MAX, allow_inf_nan=False)]


class PredictionRequest(BaseModel):
    date: Annotated[Date, BeforeValidator(_iso_date_string)] = Field(
        ...,
        examples=["1991-01-01"],
        description="Target date as a YYYY-MM-DD string. Other formats and numbers return 422.",
    )

    recent_temps: list[Temperature] = Field(
        ...,
        min_length=30,
        max_length=366,
        description=(
            "Recent daily min temps in deg C, newest first. Index 0 = yesterday. "
            f"30 to 366 finite values, each between {TEMP_MIN:g} and {TEMP_MAX:g}; "
            "only the first 30 are used."
        ),
    )


class PredictionResponse(BaseModel):
    date: str
    prediction: float
