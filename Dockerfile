FROM pytorch/pytorch:2.10.0-cuda13.0-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies (ffmpeg, libsndfile for audio)
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser -m appuser

WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir --break-system-packages -r requirements.txt

# Copy application code
COPY . .

# Create directories and set permissions
RUN mkdir -p /models /app/outputs && \
    chown -R appuser:appuser /app /models && \
    chmod -R 777 /app/outputs

# Switch to non-root user
USER appuser

# Expose Gradio port
EXPOSE 7860

ENTRYPOINT ["python3"]
CMD ["webui.py"]
