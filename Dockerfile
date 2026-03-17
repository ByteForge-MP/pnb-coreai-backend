FROM python:3.10-slim

WORKDIR /app

# Copy requirements first
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy full code
COPY . .

# Run server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]