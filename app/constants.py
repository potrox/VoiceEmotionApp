"""Общие параметры аудио, признаков и поддерживаемых эмоций."""

from __future__ import annotations

FEATURE_SCHEMA_VERSION = 4

APP_NAME = "VoiceEmotionApp"
APP_VERSION = "1.0.0"

EMOTIONS = ["joy", "sadness", "anger", "surprise", "calm", "disgust", "fear"]

EMOTION_RU = {
    "joy": "радость",
    "sadness": "грусть",
    "anger": "гнев",
    "surprise": "удивление",
    "calm": "спокойствие",
    "disgust": "отвращение",
    "fear": "страх",
}

EMOTION_EN_TO_CANONICAL = {
    "joy": "joy", "happy": "joy", "happiness": "joy", "positive": "joy", "радость": "joy", "счастье": "joy",
    "sadness": "sadness", "sad": "sadness", "sorrow": "sadness", "грусть": "sadness", "печаль": "sadness",
    "anger": "anger", "angry": "anger", "rage": "anger", "гнев": "anger", "злость": "anger", "сердитость": "anger",
    "surprise": "surprise", "surprised": "surprise", "удивление": "surprise",
    "calm": "calm", "neutral": "calm", "normal": "calm", "спокойствие": "calm", "нейтрально": "calm", "нейтральная": "calm", "нейтральный": "calm",
    "disgust": "disgust", "disgusted": "disgust", "отвращение": "disgust",
    "fear": "fear", "afraid": "fear", "scared": "fear", "страх": "fear", "испуг": "fear",
}

CSV_REQUIRED_LOGICAL_COLUMNS = ["file_path", "emotion"]
CSV_COLUMN_ALIASES = {
    "file_path": ["file_path", "path", "audio_path", "filepath", "filename", "file", "wav", "audio", "путь", "файл"],
    "original_path": ["original_path", "source_file_path", "raw_path", "исходный_файл", "путь_исходной_записи"],
    "emotion": ["emotion", "label", "class", "target", "эмоция", "метка", "класс"],
    "speaker_id": ["speaker_id", "speaker", "actor", "диктор", "ид_диктора"],
    "gender": ["gender", "sex", "пол"],
    "age_group": ["age_group", "age", "возраст", "возрастная_группа"],
    "recorded_at": ["recorded_at", "date", "record_date", "дата"],
    "duration": ["duration", "длительность"],
    "sample_rate": ["sample_rate", "sr", "частота_дискретизации"],
    "source": ["source", "источник"],
    "text": ["text", "phrase", "transcript", "фраза", "текст"],
    "language": ["language", "lang", "язык"],
    "quality": ["quality", "качество"],
}

AGE_GROUPS = ["до 18", "18-25", "26-35", "36-50", "старше 50"]
GENDERS = ["не указано", "мужской", "женский"]
LANGUAGES = ["ru", "en"]

STANDARD_PHRASES = [
    "Сегодня обычный день.",
    "Я читаю этот текст вслух.",
    "На столе лежит книга.",
    "Сейчас я произнесу короткую фразу.",
    "Погода за окном меняется.",
    "Мне нужно записать несколько предложений.",
    "Эта фраза используется для проверки голоса.",
    "Я говорю спокойно и разборчиво.",
    "В комнате достаточно тихо.",
    "Запись должна получиться чистой.",
    "Компьютер обрабатывает аудиосигнал.",
    "Модель анализирует особенности речи.",
    "Каждая запись сохраняется в датасет.",
    "Голос может звучать по-разному.",
    "Интонация помогает передавать состояние.",
    "Фраза не должна подсказывать эмоцию смыслом.",
    "Я стараюсь говорить с одинаковой громкостью.",
    "Длина записи составляет несколько секунд.",
    "После записи можно прослушать результат.",
    "Данные нужны для дальнейшего обучения.",
    "Система сравнит несколько моделей.",
    "Аудиофайл будет приведён к единому формату.",
    "Некоторые признаки описывают высоту голоса.",
    "Другие признаки связаны с энергией речи.",
    "Результат обучения зависит от качества данных.",
    "Важно записать все выбранные эмоции.",
    "Паузы между словами должны быть естественными.",
    "Эта запись станет частью эксперимента.",
    "После обработки появится новый файл.",
    "Я завершаю произнесение стандартной фразы.",
]
