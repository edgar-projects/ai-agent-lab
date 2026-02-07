from pydantic import BaseModel, Field
from typing import Literal

class Classification(BaseModel):
    label: Literal["company","person","place","product","organization","concept","other"]
    confidence: float = Field(ge=0.0, le=1.0)