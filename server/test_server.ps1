$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Get-EnvValue {
    param([string]$Name)
    $line = Get-Content ".env" | Where-Object { $_ -match "^$Name=" } | Select-Object -First 1
    if (-not $line) {
        throw "$Name отсутствует в .env"
    }
    return $line.Substring($Name.Length + 1).Trim()
}

$baseUrl = "http://127.0.0.1:8000"
$adminKey = Get-EnvValue "ADMIN_API_KEY"
$adminHeaders = @{ "X-Admin-Key" = $adminKey }

$health = Invoke-RestMethod -Method Get -Uri "$baseUrl/health"
$codeResponse = Invoke-RestMethod -Method Post -Uri "$baseUrl/api/v1/admin/registration-codes" -Headers $adminHeaders -ContentType "application/json" -Body (@{ expires_minutes = 30 } | ConvertTo-Json)

$registrationBody = @{
    code = $codeResponse.code
    display_name = "Тестовый пользователь"
    device_name = "Тестовое устройство"
    timezone_name = "Europe/Moscow"
    client_version = "0.1.0-test"
} | ConvertTo-Json

$registration = Invoke-RestMethod -Method Post -Uri "$baseUrl/api/v1/client/register" -ContentType "application/json" -Body $registrationBody
$deviceHeaders = @{ Authorization = "Bearer $($registration.token)" }

$heartbeatBody = @{
    client_version = "0.1.0-test"
    model_version = "test-model-1"
    model_sha256 = ("0" * 64)
    monitoring_active = $true
} | ConvertTo-Json

Invoke-RestMethod -Method Post -Uri "$baseUrl/api/v1/client/heartbeat" -Headers $deviceHeaders -ContentType "application/json" -Body $heartbeatBody | Out-Null

$events = @()
for ($index = 0; $index -lt 6; $index++) {
    $events += @{
        message_id = [guid]::NewGuid().ToString()
        occurred_at = (Get-Date).ToUniversalTime().AddSeconds(-1 * (30 - ($index * 5))).ToString("o")
        timezone_name = "Europe/Moscow"
        segment_duration_sec = 5.0
        speech_duration_sec = 5.0
        predicted_emotion = "joy"
        confidence = 0.91
        probabilities = @{
            joy = 0.70
            sadness = 0.05
            anger = 0.05
            surprise = 0.05
            calm = 0.10
            disgust = 0.025
            fear = 0.025
        }
        quality = @{
            signal_to_noise = 18.0
            speech_ratio = 0.85
        }
        playback_active = $false
        output_is_headphones = $true
        filtered_before_send_count = 0
        model_version = "test-model-1"
        model_sha256 = ("0" * 64)
        client_version = "0.1.0-test"
        processing_time_ms = 120.0
    }
}

$batchBody = @{ events = $events } | ConvertTo-Json -Depth 8
$batchResult = Invoke-RestMethod -Method Post -Uri "$baseUrl/api/v1/client/events/batch" -Headers $deviceHeaders -ContentType "application/json" -Body $batchBody
Start-Sleep -Seconds 5
$stats = Invoke-RestMethod -Method Get -Uri "$baseUrl/api/v1/admin/stats?user_id=$($registration.user_id)&period_type=hour" -Headers $adminHeaders

[pscustomobject]@{
    Health = $health.status
    UserId = $registration.user_id
    DeviceId = $registration.device_id
    AcceptedEvents = $batchResult.accepted
    DuplicateEvents = $batchResult.duplicates
    RejectedEvents = $batchResult.rejected
    HourlyAggregates = @($stats).Count
}
