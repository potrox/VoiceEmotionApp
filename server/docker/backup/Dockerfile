FROM postgres:17-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends openssl && rm -rf /var/lib/apt/lists/*

COPY backup.sh /scripts/backup.sh
RUN chmod 755 /scripts/backup.sh

CMD ["/bin/sh", "/scripts/backup.sh"]
