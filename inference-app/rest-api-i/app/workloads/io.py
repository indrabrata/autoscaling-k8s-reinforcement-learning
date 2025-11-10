
import asyncio
import time

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..models import Order


async def fetch_orders(session: AsyncSession, limit: int, offset: int):
    q = select(Order).order_by(Order.created_at.desc()).limit(limit).offset(offset)
    res = await session.execute(q)
    rows = res.scalars().all()
    total = await session.scalar(select(func.count()).select_from(Order))
    return rows, int(total or 0)


async def get_orders(session: AsyncSession, limit: int, offset: int):
    t0 = time.perf_counter()
    rows, total = await fetch_orders(session, limit, offset)
    dt = (time.perf_counter() - t0) * 1000.0
    payload = [{
        "id": r.id, "user_id": r.user_id, "amount": r.amount,
        "status": r.status, "created_at": r.created_at.isoformat()
    } for r in rows]
    return {"total": total, "items": payload, "elapsed_ms": dt}
