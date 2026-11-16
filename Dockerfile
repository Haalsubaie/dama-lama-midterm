
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY model.bin .
COPY predict.py .

EXPOSE 9696

CMD ["waitress-serve", "--listen=0.0.0.0:9696", "predict:app"]
