FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /srv

RUN groupadd --system voiceemotion && useradd --system --gid voiceemotion --home /srv voiceemotion

COPY requirements.txt /srv/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r /srv/requirements.txt

COPY . /srv
RUN mkdir -p /data/models /data/logs && chown -R voiceemotion:voiceemotion /srv /data

USER voiceemotion

CMD ["uvicorn", "server_app.main:app", "--host", "0.0.0.0", "--port", "8000"]
