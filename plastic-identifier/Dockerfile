# syntax=docker/dockerfile:1
FROM python:3.9-alpine

COPY ./scripts/env.txt requirements.txt

RUN pip install -r requirements.txt

COPY ./scripts/reader_agent.py .

CMD ["python", "-u", "reader_agent.py"]
