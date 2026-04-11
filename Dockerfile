FROM python:3.11-slim

# HuggingFace Spaces runs as non-root user 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Set working directory
WORKDIR /home/user/app

# Install dependencies — use root requirements.txt (not server/)
COPY --chown=user requirements.txt requirements.txt
RUN pip install --no-cache-dir --user -r requirements.txt

# Copy entire project
COPY --chown=user . /home/user/app/worklens_env

# Set Python path so worklens_env is importable
ENV PYTHONPATH="/home/user/app"

# HuggingFace Spaces uses port 7860
ENV PORT=7860
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')"

# Run the server
CMD ["bash", "-c", "python -m uvicorn worklens_env.server.app:app --host 0.0.0.0 --port 7860 & sleep 2 && python worklens_env/inference.py"]