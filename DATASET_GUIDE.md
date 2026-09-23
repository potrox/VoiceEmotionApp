# Датасет для честной оценки VoiceEmotionApp

## Рекомендация

Использовать **Dusha crowd** — русскоязычную часть открытого корпуса Salute/Sber для распознавания эмоций в речи.

Официальные источники:

- документация Sber: <https://developers.sber.ru/docs/ru/salutespeech/guides/datasets>
- репозиторий и описание: <https://github.com/salute-developers/golos/tree/master/dusha>
- официальный архив `crowd.tar`: <https://cdn.chatwm.opensmodel.sberdevices.ru/dusha/crowd.tar>
- статья Interspeech 2023: <https://www.isca-archive.org/interspeech_2023/kondratenko23_interspeech.pdf>

Полный корпус содержит 303 963 записи, 346,6 часа и 8308 дикторов. Доступная для скачивания crowd-часть содержит 201 850 файлов, 255,7 часа и 2068 дикторов. Классы: happiness/positive, sadness, anger и neutral.

## Почему четыре класса

Это осознанное упрощение:

- все четыре класса представлены в одном крупном русскоязычном источнике;
- не возникает зависимости «эмоция = источник датасета»;
- больше дикторов в каждом классе позволяет честно отделить test по людям;
- задача лучше соответствует реальному использованию русскоязычного приложения;
- обучение дешевле и устойчивее, чем семь классов на небольшом частном наборе.

Соответствие классам приложения:

| Dusha | VoiceEmotionApp |
|---|---|
| positive / happiness | joy |
| sadness | sadness |
| anger | anger |
| neutral | calm |

Классы surprise, disgust и fear в этом эксперименте исключаются. Их нельзя добавлять из другого датасета только ради количества классов: модель сможет выучить источник записи вместо эмоции.

## Зафиксированная версия эксперимента

Из полного архива подготовлена сбалансированная выборка из 15 000 WAV-файлов:

- train: 12 000 записей, по 3000 на класс;
- test: 3000 записей, по 750 на класс;
- не более 12 train-записей одного диктора на класс и 6 test-записей;
- 1766 train-дикторов и 188 test-дикторов;
- пересечение дикторов train/test: 0;
- confidence Dawid–Skene >= 0,9;
- seed = 42;
- grouped CV только на обучающих дикторах.

Файл `dusha_15k.csv` сохраняет поле `dataset_split`, поэтому приложение использует официальный split корпуса, а не создаёт новый holdout.

## Подготовка CSV

Скрипт получает сырые аннотации из официальных TSV/JSONL, агрегирует их и извлекает из `crowd.tar` только выбранные аудиофайлы:

```powershell
python scripts/prepare_dusha_archive.py `
  --archive "C:\Downloads\crowd.tar" `
  --metadata-dir "C:\Datasets\Dusha\metadata" `
  --output-root "C:\Datasets\Dusha_15k" `
  --output-csv "C:\Datasets\Dusha_15k\dusha_15k.csv" `
  --train-per-class 3000 `
  --test-per-class 750 `
  --seed 42
```

Сводка параметров и распределений сохраняется рядом с CSV в `dusha_15k.summary.json`.

## Протокол метрик

- Один диктор может находиться только в одном из наборов: train или test.
- Подбор модели и параметров выполняется только grouped CV внутри train.
- Test не участвует в выборе лучшей модели.
- Основная метрика — macro F1.
- Смещения многоклассового решающего правила настраиваются только по grouped OOF-прогнозам train.
- Дополнительно публикуются balanced accuracy, macro precision, macro recall, weighted F1, ROC-AUC, PR-AUC, confusion matrix, per-class метрики и 95% bootstrap-интервалы по дикторам.
- Accuracy не используется как основная метрика при дисбалансе классов.
