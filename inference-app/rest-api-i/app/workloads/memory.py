import time
from collections import defaultdict
from typing import Any, Dict, List

from ..schemas import TransactionItem


def analyze(trans: List[TransactionItem], expand_factor: int = 1, heavy_agg: bool = False):
    t0 = time.perf_counter()
    
    expanded: List[TransactionItem] = trans * expand_factor
    
    totals: Dict[str, float] = defaultdict(float)
    for t in expanded:
        totals[t.category] += t.amount
    
    per_user: Dict[int, float] = defaultdict(float)
    for t in expanded:
        per_user[t.user_id] += t.amount
    
    category_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for t in expanded:
        category_history[t.category].append({
            "user_id": t.user_id,
            "amount": t.amount,
            "timestamp": getattr(t, 'timestamp', None),
            "metadata": {"category": t.category, "amount_squared": t.amount ** 2}
        })
    
    user_history: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for t in expanded:
        user_history[t.user_id].append({
            "category": t.category,
            "amount": t.amount,
            "timestamp": getattr(t, 'timestamp', None),
            "metadata": {"user_id": t.user_id, "amount_cubed": t.amount ** 3}
        })
    
    transaction_matrix = []
    for i, t in enumerate(expanded):
        row = [t.amount] * 100
        transaction_matrix.append(row)
    
    top_users = sorted(
        ({"user_id": uid, "total": amt, "count": len([t for t in expanded if t.user_id == uid])} 
         for uid, amt in per_user.items()),
        key=lambda x: x["total"], reverse=True
    )[:10]
    
    if heavy_agg:
        buckets = defaultdict(list)
        for t in expanded:
            buckets[(t.category, t.user_id)].append(t.amount)
        
        _means = {k: (sum(v)/len(v)) for k, v in buckets.items()}
        _stds = {k: (sum((x - _means[k])**2 for x in v) / len(v))**0.5 
                 for k, v in buckets.items()}
        
        user_amounts = list(per_user.values())
        correlation_data = []
        for i in range(min(len(user_amounts), 1000)):
            correlation_row = [user_amounts[i] * user_amounts[j] 
                             for j in range(min(len(user_amounts), 1000))]
            correlation_data.append(correlation_row)
    
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    
    base_trans_size = len(expanded) * 200 
    history_size = sum(len(v) * 150 for v in category_history.values())
    user_history_size = sum(len(v) * 150 for v in user_history.values())
    matrix_size = len(transaction_matrix) * 100 * 8 
    
    peak_mb = (base_trans_size + history_size + user_history_size + matrix_size) / (1024*1024)
    
    return {
        "num_input": len(trans),
        "expanded": len(expanded),
        "totals_by_category": dict(totals),
        "top_users": top_users,
        "elapsed_ms": elapsed_ms,
        "peak_memory_est_mb": peak_mb,
        "structures_created": {
            "category_history": len(category_history),
            "user_history": len(user_history),
            "transaction_matrix_rows": len(transaction_matrix)
        }
    }
