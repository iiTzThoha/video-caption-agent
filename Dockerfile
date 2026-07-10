FROM python:3.11-slim
WORKDIR /app
COPY pipeline.py .
RUN pip install --no-cache-dir requests
CMD ["python", "pipeline.py"]