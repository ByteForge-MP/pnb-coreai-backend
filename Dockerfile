FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_DEFAULT_TIMEOUT=1000

WORKDIR /app

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

# 1. Copy only requirements first (cache optimization)
COPY requirements.txt .

# 2. Install PyTorch from a selectable index.
# Default is CPU-only. For NVIDIA GPU builds, pass for example:
# --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121
RUN pip install --no-cache-dir --index-url ${TORCH_INDEX_URL} torch

# 3. Install app dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy ONLY required code (NOT everything)
COPY app/ app/
COPY abbreviations.json .

# If you have other folders, add explicitly:
# COPY config/ config/

EXPOSE 8000

CMD ["python", "-m", "app.main", "--ollama", "true"]
