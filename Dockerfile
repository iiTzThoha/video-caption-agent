FROM python:3.11-slim
WORKDIR /app
COPY pipeline.py .
RUN pip install --no-cache-dir requests
ENV FIREWORKS_API_KEY=API_here
CMD ["python", "pipeline.py"]
