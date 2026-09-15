#!/bin/sh
set -eu

mkdir -p /backups/daily /backups/weekly

while true; do
  timestamp="$(date -u +%Y%m%d_%H%M%S)"
  daily="/backups/daily/voiceemotion_${timestamp}.dump"
  encrypted="${daily}.enc"
  pg_dump -h db -U "$POSTGRES_USER" -d "$POSTGRES_DB" -Fc -f "$daily"
  openssl enc -aes-256-cbc -salt -pbkdf2 -pass env:BACKUP_ENCRYPTION_KEY -in "$daily" -out "$encrypted"
  rm -f "$daily"
  find /backups/daily -type f -name '*.enc' -mtime +14 -delete
  weekday="$(date -u +%u)"
  if [ "$weekday" = "7" ]; then
    cp "$encrypted" "/backups/weekly/$(basename "$encrypted")"
  fi
  find /backups/weekly -type f -name '*.enc' -mtime +60 -delete
  sleep 86400
done
