FROM python:3.10-slim

WORKDIR /app

# Install system dependencies (needed for some python packages like numpy/pandas sometimes, and git)
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml .
COPY src/ src/
COPY benchmarks/ benchmarks/
COPY examples/ examples/
COPY tests/ tests/
COPY README.md .

# Install dependencies (Using pip directly since we already have pyproject.toml)
# We can use poetry or just pip install . if setup.py exists. 
# Since we only have pyproject.toml, we'll install poetry to export requirements or install directly.
# Simplest for this D4 task:
RUN pip install poetry && poetry config virtualenvs.create false && poetry install --no-interaction --no-ansi

# Default command
ENTRYPOINT ["llm-eval"]
CMD ["--help"]
