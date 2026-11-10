FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p results

CMD ["sh", "-c", "python scripts/demo_inference.py --images demo_test/images --out results && python scripts/eval_on_demo.py --data demo_test --out results"]