param(
    [int]$ExpiresMinutes = 30
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$envLines = Get-Content ".env"
$adminLine = $envLines | Where-Object { $_ -match '^ADMIN_API_KEY=' } | Select-Object -First 1
if (-not $adminLine) {
    throw "ADMIN_API_KEY отсутствует в .env"
}
$adminKey = $adminLine.Substring("ADMIN_API_KEY=".Length)
$body = @{ expires_minutes = $ExpiresMinutes } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/api/v1/admin/registration-codes" -Headers @{ "X-Admin-Key" = $adminKey } -ContentType "application/json" -Body $body
