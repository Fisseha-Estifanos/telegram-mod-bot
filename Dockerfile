FROM python:3.12-slim

WORKDIR /app

# Install uv
RUN pip install --no-cache-dir uv

# Copy project files
COPY pyproject.toml .
COPY src/ src/
COPY scripts/ scripts/

# Install dependencies using uv
RUN uv pip install --system .

CMD ["python", "-m", "src.main"]
