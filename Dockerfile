# 1. Используем официальный базовый образ Python. "slim" версия меньше по размеру.
FROM python:3.9-slim

# 2. Устанавливаем переменные окружения для Python
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# 3. Устанавливаем рабочую директорию внутри контейнера
WORKDIR /app

# 4. Создаем не-рутового пользователя для безопасности
RUN addgroup --system app && adduser --system --ingroup app app

# 5. Копируем файл с зависимостями и устанавливаем их.
# Этот шаг выполняется отдельно от копирования кода для использования кэширования Docker.
# Если зависимости не меняются, этот слой не будет пересобираться.
COPY --chown=app:app requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 6. Копируем остальной код нашего приложения в рабочую директорию
COPY --chown=app:app . .

# 7. Переключаемся на не-рутового пользователя
USER app

# 8. Настройка переменных окружения для coverage (если нужно)
ENV COVERAGE_FILE=/tmp/.coverage

# 9. Команда по умолчанию, которая будет выполняться при запуске контейнера
CMD ["python", "train.py", "--config", "configs/base_config.yaml"]
