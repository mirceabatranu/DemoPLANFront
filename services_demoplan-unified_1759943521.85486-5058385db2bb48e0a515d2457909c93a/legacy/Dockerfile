# DEMOPLAN Phase 1 - FIXED Dockerfile (Removed problematic ML dependencies)
FROM python:3.11-slim

# Set environment variables to avoid debconf issues
ENV DEBIAN_FRONTEND=noninteractive
ENV TERM=xterm

# Install ONLY essential system dependencies (removed problematic ML libs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set working directory
WORKDIR /app

# Set Python environment variables for ML
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Phase 1 ML Integration environment
ENV PHASE=1
ENV ML_ENABLED=true
ENV SAFETY_FRAMEWORK_ENABLED=true
ENV ROMANIAN_CONSTRUCTION_MODE=true

# ML Configuration
ENV ML_CONFIDENCE_ADJUSTMENT=0.5
ENV CONFIDENCE_MIN_CAP=0.2
ENV CONFIDENCE_MAX_CAP=0.95
ENV PATTERN_MATCHING_ENABLED=true
ENV MIN_SIMILARITY_THRESHOLD=0.6

# Session Management
ENV SESSION_TIMEOUT_HOURS=24
ENV CHAT_SESSION_TIMEOUT=3600
ENV MAX_CONCURRENT_SESSIONS=50

# Agent Confidence Weights
ENV FLOORPLAN_AGENT_WEIGHT=0.25
ENV VALIDATION_AGENT_WEIGHT=0.15
ENV ESTIMATION_AGENT_WEIGHT=0.40
ENV OFFER_COMPOSER_WEIGHT=0.20

# Training Data Configuration
ENV TRAINING_DATA_ENABLED=true
ENV MAX_TRAINING_FILE_SIZE_MB=50
ENV TRAINING_BATCH_SIZE=10
ENV PATTERN_UPDATE_FREQUENCY=on_upload

# Learning Engine Weights
ENV COST_ACCURACY_WEIGHT=0.30
ENV TIMELINE_ACCURACY_WEIGHT=0.25
ENV CLIENT_SATISFACTION_WEIGHT=0.20
ENV COMPLEXITY_PREDICTION_WEIGHT=0.15

# Firestore Collections for ML
ENV FIRESTORE_PREFIX=demoplan_
ENV FIRESTORE_CHAT_SESSIONS=engineer_chat_sessions
ENV FIRESTORE_TRAINING_DATA=ml_training_data
ENV FIRESTORE_PROJECT_LEARNING=project_learning_outcomes
ENV FIRESTORE_PATTERN_TEMPLATES=ml_pattern_templates

# Copy requirements and install dependencies
COPY requirements.txt .

# Install Python dependencies (Python handles ML libs better than system packages)
RUN pip install --no-cache-dir --upgrade pip --root-user-action=ignore && \
    pip install --no-cache-dir -r requirements.txt --root-user-action=ignore

# Copy application code in proper order
COPY config/ ./config/
COPY src/ ./src/
COPY data/ ./data/
COPY main_runner.py .

# Create directories and set up user
RUN mkdir -p \
    logs/ml_logs \
    logs/agent_logs \
    logs/session_logs \
    logs/pattern_logs \
    data/patterns \
    data/training \
    && useradd -m -u 1000 appuser \
    && chown -R appuser:appuser /app

# Switch to non-root user for security
USER appuser

# Enhanced health check for ML components
HEALTHCHECK --interval=30s --timeout=30s --start-period=90s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Expose port
EXPOSE 8080

# Start command with the new main runner
CMD ["python", "main_runner.py"]

# Start command with ML-enhanced main API
CMD ["python", "-m", "uvicorn", "src.api.main_api:app", "--host", "0.0.0.0", "--port", "8080"]