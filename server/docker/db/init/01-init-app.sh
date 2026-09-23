#!/bin/sh
set -eu

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" \
  --set=app_user="$APP_DB_USER" \
  --set=app_password="$APP_DB_PASSWORD" <<'SQL'
SELECT format('CREATE ROLE %I LOGIN PASSWORD %L NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION', :'app_user', :'app_password')
WHERE NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = :'app_user')\gexec
SELECT format('GRANT CONNECT ON DATABASE %I TO %I', current_database(), :'app_user')\gexec
SELECT format('CREATE SCHEMA IF NOT EXISTS app AUTHORIZATION %I', :'app_user')\gexec
REVOKE CREATE ON SCHEMA public FROM PUBLIC;
SELECT format('ALTER ROLE %I IN DATABASE %I SET search_path = app, public', :'app_user', current_database())\gexec
SQL
