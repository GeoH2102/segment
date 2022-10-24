FROM python:3.10.6-slim-bullseye

WORKDIR /segment-app

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY models/ models/
COPY Segment/ Segment/
COPY static/ static/
COPY templates/ templates/
COPY app.py app.py

CMD ["python3", "-m", "flask", "run", "--host=0.0.0.0"]