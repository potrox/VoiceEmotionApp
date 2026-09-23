# VoiceEmotionServer

Серверный модуль принимает события распознавания и не принимает аудиофайлы. Он необязателен для desktop-распознавания. Схема событий поддерживает четыре класса текущей модели и семь классов прежних клиентов.

## Первый запуск

1. Создайте локальный файл `server/.env`.
2. Задайте параметры PostgreSQL: `POSTGRES_DB`, `POSTGRES_USER`, `POSTGRES_PASSWORD`, `APP_DB_USER`, `APP_DB_PASSWORD` и `DATABASE_URL`.
3. Задайте секреты длиной не менее 32 символов: `ADMIN_API_KEY`, `TOKEN_SECRET`, `MODEL_SIGNING_KEY`, `DATA_ENCRYPTION_KEY` и `BACKUP_ENCRYPTION_KEY`.
4. При необходимости задайте `SERVER_MODELS_DIR`, сроки хранения, интервалы фоновых задач, `API_BIND_ADDRESS` и `LOG_LEVEL`.
5. Выполните `powershell -ExecutionPolicy Bypass -File .\start_server.ps1`.
6. Проверьте `http://127.0.0.1:8000/health` и откройте `http://127.0.0.1:8000/docs` для тестирования API.

В desktop-приложении откройте вкладку «Сервер»: по умолчанию она использует папку `server` этого репозитория и адрес `http://127.0.0.1:8000`. Удалённый адрес административного API допускается только через HTTPS, чтобы не передавать ключ по открытому HTTP. События desktop-приложение автоматически не отправляет; их принимают зарегистрированные API-клиенты.

## Основные маршруты

- `GET /health`
- `POST /api/v1/admin/registration-codes`
- `POST /api/v1/client/register`
- `POST /api/v1/client/heartbeat`
- `POST /api/v1/client/switch-user`
- `POST /api/v1/client/events/batch`
- `GET /api/v1/client/models/current`
- `GET /api/v1/client/models/{version}/download`
- `POST /api/v1/admin/models`
- `POST /api/v1/admin/models/{version}/activate`
- `GET /api/v1/admin/users`
- `GET /api/v1/admin/devices`
- `GET /api/v1/admin/stats`

## Локальная проверка

По умолчанию API доступен только по `127.0.0.1:8000`. Не меняйте `API_BIND_ADDRESS` на `0.0.0.0` до настройки HTTPS или защищённой локальной сети.

## Полная автоматическая проверка

После запуска контейнеров выполните:

```powershell
powershell -ExecutionPolicy Bypass -File .\test_server.ps1
```

Быстрые Python-тесты схемы и агрегации без Docker (после установки `server/requirements.txt`):

```powershell
cd server
python -m unittest discover -s tests -v
```

Скрипт создаёт одноразовый код, регистрирует тестовое устройство, отправляет шесть событий общей длительностью 30 секунд и проверяет формирование часового агрегата.
