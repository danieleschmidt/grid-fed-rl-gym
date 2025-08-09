# Multi-stage Docker build for Grid-Fed-RL-Gym
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Create non-root user for security
RUN groupadd --gid 1000 gridrl && \
    useradd --uid 1000 --gid gridrl --shell /bin/bash --create-home gridrl

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    gfortran \
    libblas-dev \
    liblapack-dev \
    libhdf5-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# ==================================
# Development stage
# ==================================
FROM base as development

# Install development dependencies
RUN pip install --upgrade pip setuptools wheel

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt requirements-dev.txt ./
RUN pip install -r requirements-dev.txt

# Copy source code
COPY . .

# Install package in development mode
RUN pip install -e ".[dev]"

# Change ownership to non-root user
RUN chown -R gridrl:gridrl /app
USER gridrl

# Expose ports
EXPOSE 8080 9090

# Development command
CMD ["python", "-m", "grid_fed_rl.cli", "--help"]

# ==================================
# Production stage  
# ==================================
FROM base as production

# Install only production dependencies
RUN pip install --upgrade pip setuptools wheel

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt ./
RUN pip install -r requirements.txt

# Copy source code (excluding development files)
COPY grid_fed_rl/ ./grid_fed_rl/
COPY setup.py pyproject.toml README.md LICENSE ./

# Install package
RUN pip install . --no-deps

# Create directories for data and logs
RUN mkdir -p /app/data /app/logs /app/cache && \
    chown -R gridrl:gridrl /app

# Switch to non-root user
USER gridrl

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import grid_fed_rl; print('OK')" || exit 1

# Expose ports
EXPOSE 8080 9090

# Production command
CMD ["python", "-m", "grid_fed_rl.cli", "serve", "--host", "0.0.0.0", "--port", "8080"]

# ==================================
# Testing stage
# ==================================
FROM development as testing

# Run tests
RUN python -m pytest tests/ -v --cov=grid_fed_rl --cov-report=html --cov-report=term

# ==================================
# Default production
# ==================================
FROM production