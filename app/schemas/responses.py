from pydantic import BaseModel

class TrainResponse(BaseModel):
    success: bool
    message: str
    processed_records: int
    total_records: int 