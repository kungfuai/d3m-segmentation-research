FROM tensorflow/tensorflow:2.2.0

WORKDIR /workspace

ENV PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y \
    dumb-init \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Install pip packages
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Switch to non-root user
RUN useradd -m appuser && chown -R appuser /workspace
USER appuser

# Copy project files
COPY . .

ENTRYPOINT ["/usr/bin/dumb-init", "--"]
CMD ["python", "-m", "src.main"]
