$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
if (-not (Test-Path ".env")) {
    throw "Создайте server/.env и заполните переменные окружения перед запуском."
}
docker compose config | Out-Null
docker compose build
docker compose up -d

docker compose ps
