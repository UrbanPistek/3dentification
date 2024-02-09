# syntax=docker/dockerfile:1
FROM python:3.9-alpine

COPY env.txt requirements.txt

RUN pip install -r requirements.txt

COPY detection_agent.py .

CMD ["python", "-u", "detection_agent.py"]
