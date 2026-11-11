FROM python:3.11-slim

# Install OpenMP (libgomp1), which LightGBM requires on Linux
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model artifacts and source code
COPY models/ ./models/
COPY src/ ./src/

EXPOSE 8000

# Start FastAPI server
CMD ["uvicorn", "src.3_api_and_agent:app", "--host", "0.0.0.0", "--port", "8000"]
