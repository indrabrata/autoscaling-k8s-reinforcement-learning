import io
import time
from typing import Any, Dict
from PIL import Image

import torch
import torchvision.transforms as T
from torchvision.models import squeezenet1_1, SqueezeNet1_1_Weights

weights = SqueezeNet1_1_Weights.DEFAULT
model = squeezenet1_1(weights=weights)
model.eval()
transform = weights.transforms()
idx2label = weights.meta["categories"]


def classify_image(file_bytes: bytes, topk: int = 5) -> Dict[str, Any]:
    t0 = time.perf_counter()
    
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    x = transform(img).unsqueeze(0)

    with torch.no_grad():
        logits = model(x)
        probs = torch.nn.functional.softmax(logits[0], dim=0)

    values, indices = torch.topk(probs, topk)
    preds = [{"label": idx2label[idx], "prob": float(val)} for val, idx in zip(values, indices)]

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return {"predictions": preds, "elapsed_ms": elapsed_ms}
