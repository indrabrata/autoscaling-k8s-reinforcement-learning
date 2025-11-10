import datetime
import json
import random

categories = ["food", "transport", "entertainment", "shopping", "bills", "salary", "health", "education"]

def random_timestamp(start, end):
    return (start + datetime.timedelta(seconds=random.randint(0, int((end - start).total_seconds())))).isoformat()

transactions = []
start_date = datetime.datetime(2023, 1, 1)
end_date = datetime.datetime(2023, 12, 31)

for i in range(100):
    transaction = {
        "id": i,
        "amount": round(random.uniform(5, 5000), 2),
        "category": random.choice(categories),
        "user_id": random.randint(1, 10),
        "ts": random_timestamp(start_date, end_date)
    }
    transactions.append(transaction)

result = {
    "transactions": transactions,
    "expand_factor": 1,
    "heavy_agg": False
}

json_output = json.dumps(result, indent=2)

output_path = ".load/seed_analyze.json"
with open(output_path, "w") as f:
    f.write(json_output)

