# Карта проекта VoiceEmotionApp

## Состав проекта

Настольное приложение находится в каталоге `app`. Оно загружает записи, извлекает
признаки, обучает модели и распознает эмоции. Каталог `server` содержит отдельный
FastAPI сервис для хранения версий моделей и результатов распознавания.

Основные модули приложения:

- [`app/ui`](../app/ui) содержит главное окно и вкладки.
- [`app/audio.py`](../app/audio.py) загружает, обрабатывает и воспроизводит звук.
- [`app/microphone.py`](../app/microphone.py) управляет потоком микрофона.
- [`app/recorder.py`](../app/recorder.py) сохраняет пользовательские записи.
- [`app/dataset.py`](../app/dataset.py) читает и проверяет CSV датасеты.
- [`app/dataset_recovery.py`](../app/dataset_recovery.py) восстанавливает CSV по файлам.
- [`app/features.py`](../app/features.py) извлекает акустические признаки.
- [`app/ml`](../app/ml) содержит обучение, оценку и сохранение моделей.
- [`app/recognizer.py`](../app/recognizer.py) выполняет распознавание.
- [`app/storage.py`](../app/storage.py) хранит историю и формирует отчеты.
- [`app/workers.py`](../app/workers.py) запускает долгие операции в потоках Qt.
- [`app/server_integration.py`](../app/server_integration.py) обращается к серверному API.

## Запуск интерфейса

1. [`main.py`](../main.py) подготавливает виртуальное окружение.
2. `main()` создает `QApplication` и `MainWindow`.
3. `MainWindow` создает восемь вкладок из модулей `app/ui`.
4. Долгие операции передаются классам из `app/workers.py`.
5. Worker отправляет интерфейсу прогресс, результат или сообщение об ошибке.

## Загрузка датасета

1. `DatasetTabMixin._select_dataset_csv()` получает путь к CSV.
2. `inspect_emotion_labels()` находит названия столбцов и неизвестные эмоции.
3. При необходимости `EmotionMappingDialog` запрашивает сопоставление меток.
4. `DatasetLoadWorker` запускает `DatasetLoader.load_csv()`.
5. `DatasetLoader` проверяет строки, пути, длительность и качество записей.
6. Готовый `DatasetLoadResult` сохраняется в `MainWindow.dataset_result`.

## Запись голоса

1. `RecordingTabMixin._record_voice()` проверяет параметры записи.
2. `MicrophoneRecorder.start()` открывает входной аудиопоток.
3. `MicrophoneRecorder.stop()` возвращает записанный сигнал.
4. `preprocess_signal()` нормализует запись и удаляет тишину.
5. `UserDatasetRecorder.save_recording()` сохраняет исходный и обработанный WAV.
6. Метаданные записи добавляются в `user_dataset.csv`.

## Обучение

1. `TrainingTabMixin._start_training()` собирает параметры формы.
2. `TrainingWorker` запускает `TrainingService.execute()`.
3. `FeatureExtractor.build_matrix()` формирует матрицу из 416 признаков.
4. `Trainer.train_all()` разделяет данные на train, validation и test.
5. Методы `_train_svm()`, `_train_random_forest()` и `_train_mlp()` обучают модели.
6. `_train_ensembles()` объединяет выбранные базовые модели.
7. Лучшая модель выбирается по macro F1 на validation.
8. `evaluate_model()` рассчитывает итоговые метрики на test.
9. `save_bundle()` сохраняет модель, преобразования и метаданные.
10. `ExperimentDB.add_experiment()` записывает результат в SQLite.
11. `write_training_report()` создает текстовый отчет.

Разбиение по `speaker_id` не допускает появления одного диктора в разных частях
выборки. Масштабирование и отбор признаков выполняются отдельно внутри каждого
фолда кросс валидации.

## Распознавание

1. `Recognizer` загружает пакет модели через `load_bundle()`.
2. `recognize_file()` читает и обрабатывает аудиофайл.
3. `FeatureExtractor` создает признаки в том же порядке, что и при обучении.
4. `transform_features()` применяет сохраненные преобразования.
5. Выбранная модель возвращает вероятности эмоций.
6. Длинная запись делится на сегменты методом `_split_signal_into_segments()`.
7. `_average_probabilities()` рассчитывает общий результат по сегментам.

## Сервер

FastAPI приложение находится в [`server/server_app`](../server/server_app).
Маршруты разделены по назначению:

- `routers/client.py` регистрирует устройства и принимает heartbeat.
- `routers/events.py` принимает результаты распознавания.
- `routers/models_router.py` отдает клиенту активную модель.
- `routers/admin.py` управляет пользователями, устройствами и моделями.
- `routers/stats.py` возвращает агрегированную статистику.

`services/aggregation_core.py` рассчитывает статистику без обращения к базе данных.
`services/aggregation.py` читает события и сохраняет результат через SQLAlchemy.
Аудиофайлы серверу не передаются.
