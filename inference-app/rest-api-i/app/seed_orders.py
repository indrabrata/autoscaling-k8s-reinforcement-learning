import asyncio
import random
from datetime import datetime, timedelta, timezone

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert

from .db import engine, Base, AsyncSessionLocal
from .models import Order

STATUSES = ["created", "paid", "shipped", "cancelled", "refunded"]


async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def seed_orders(n: int = 5000, batch_size: int = 2000):
    async with AsyncSessionLocal() as session:
        count = (await session.execute(text("SELECT COUNT(*) FROM orders"))).scalar() or 0
        if count >= n:
            print(f"Orders already seeded ({count} rows). Skipping.")
            return

        now = datetime.now(timezone.utc)
        rows = []
        for i in range(n):
            dt = now - timedelta(minutes=random.randint(0, 60 * 24 * 30))
            rows.append({
                "user_id": random.randint(1, 2000),
                "amount": round(random.uniform(5, 500), 2),
                "status": random.choice(STATUSES),
                "meta": {"source": random.choice(["web", "mobile", "ads"])},
                "created_at": dt,
            })

            if len(rows) >= batch_size:
                await bulk_insert(session, rows)
                rows.clear()

        if rows:
            await bulk_insert(session, rows)

        print(f"✅ Seeded {n} orders.")


async def bulk_insert(session: AsyncSession, rows):
    stmt = insert(Order).values(rows)
    await session.execute(stmt)
    await session.commit()


async def main():
    await init_db()
    await seed_orders()


if __name__ == "__main__":
    asyncio.run(main())
