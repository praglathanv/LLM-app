FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=100 --retries 5 -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "api:api", "--host", "0.0.0.0", "--port", "8000"]