FROM pytorch/pytorch:2.10.0-cuda13.0-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Build arguments for user mapping
ARG UID=1000
ARG GID=1000

# Install system dependencies (ffmpeg, libsndfile for audio)
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user with specified UID/GID
RUN if ! getent group ${GID} > /dev/null 2>&1; then \
        groupadd -g ${GID} appuser; \
    else \
        existing_group=$(getent group ${GID} | cut -d: -f1); \
        groupmod -n appuser ${existing_group}; \
    fi && \
    if ! getent passwd ${UID} > /dev/null 2>&1; then \
        useradd -u ${UID} -g appuser -m appuser; \
    else \
        existing_user=$(getent passwd ${UID} | cut -d: -f1); \
        usermod -l appuser -d /home/appuser -m ${existing_user}; \
    fi

WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir --break-system-packages -r requirements.txt

# Copy application code
COPY . .

# Create directories and set ownership
RUN mkdir -p /models /app/outputs /app/samples && \
    chown -R appuser:appuser /app /models

# Switch to non-root user
USER appuser

# Expose Gradio port
EXPOSE 7860

CMD ["python3", "webui.py"]
