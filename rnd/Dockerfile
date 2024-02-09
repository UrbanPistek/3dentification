# syntax=docker/dockerfile:1
FROM python:3.9-alpine

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

COPY app.py .

CMD ["python", "-u", "app.py"]
