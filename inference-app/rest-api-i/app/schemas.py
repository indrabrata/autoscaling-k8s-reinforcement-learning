from pydantic import BaseModel, Field
from typing import List, Literal, Optional, Dict, Any

class PredictionItem(BaseModel):
    label: str
    prob: float

class PredictResponse(BaseModel):
    result: List[PredictionItem]
    elapsed_ms: float

class TransactionItem(BaseModel):
    id: int
    amount: float
    category: str
    user_id: int
    ts: str

class AnalyzeRequest(BaseModel):
    transactions: List[TransactionItem]
    expand_factor: int
    heavy_agg: bool = False

class AnalyzeResponse(BaseModel):
    num_input: int
    expanded: int
    totals_by_category: Dict[str, float]
    top_users: List[Dict[str, Any]]
    elapsed_ms: float
    peak_memory_est_mb: Optional[float] = None

class OrderOut(BaseModel):
    id: int
    user_id: int
    amount: float
    status: str
    created_at: str

class OrdersResponse(BaseModel):
    total: int
    items: List[OrderOut]
    elapsed_ms: float
