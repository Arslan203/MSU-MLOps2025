#!/bin/bash
# Pre-commit hook для запуска тестов перед коммитом
# Использует Docker для запуска проверок (как в CI)
# Установка: cp scripts/pre-commit-hook.sh .git/hooks/pre-commit && chmod +x .git/hooks/pre-commit

echo "🔍 Запуск проверок перед коммитом (как в CI)..."

# Проверяем наличие Docker
if ! command -v docker &> /dev/null; then
    echo "⚠️  Docker не найден. Пропускаем локальные проверки."
    echo "💡 Проверки будут выполнены в GitHub Actions CI."
    exit 0
fi

# Проверяем что Docker работает
if ! docker info &> /dev/null; then
    echo "⚠️  Docker не запущен. Пропускаем локальные проверки."
    echo "💡 Проверки будут выполнены в GitHub Actions CI."
    exit 0
fi

# Имя образа
IMAGE_NAME="recipe-ranker-app"

# Определяем корень проекта (git root)
if command -v git &> /dev/null; then
    PROJECT_DIR="$(git rev-parse --show-toplevel 2>/dev/null)"
fi

# Если git не доступен, используем путь относительно скрипта
if [ -z "$PROJECT_DIR" ] || [ ! -d "$PROJECT_DIR" ]; then
    # Hook находится в .git/hooks/, нужно подняться на 2 уровня
    HOOK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_DIR="$(cd "$HOOK_DIR/../.." && pwd)"
fi

# Проверяем что мы в правильной директории
if [ ! -f "$PROJECT_DIR/Dockerfile" ] || [ ! -d "$PROJECT_DIR/src" ]; then
    echo "❌ Не удалось найти корень проекта. Убедитесь что вы в git репозитории."
    exit 1
fi

# Собираем образ если его нет
if ! docker image inspect "$IMAGE_NAME" &> /dev/null; then
    echo "📦 Сборка Docker образа $IMAGE_NAME..."
    cd "$PROJECT_DIR"
    docker build -t "$IMAGE_NAME" . > /dev/null 2>&1
    if [ $? -ne 0 ]; then
        echo "❌ Ошибка сборки Docker образа"
        exit 1
    fi
fi

cd "$PROJECT_DIR"

# Проверка форматирования с black (как в CI)
echo "📝 Проверка форматирования кода (black)..."
docker run --rm -v "$PROJECT_DIR:/app" -w /app "$IMAGE_NAME" black --check src tests
BLACK_STATUS=$?

# Проверка линтера - первая часть (критические ошибки)
echo "🔎 Проверка линтера flake8 (критические ошибки)..."
docker run --rm -v "$PROJECT_DIR:/app" -w /app "$IMAGE_NAME" \
    flake8 src tests --count --select=E9,F63,F7,F82 --show-source --statistics
FLAKE8_STATUS=$?

# Проверка линтера - вторая часть (полная проверка)
echo "🔎 Проверка линтера flake8 (полная проверка)..."
docker run --rm -v "$PROJECT_DIR:/app" -w /app "$IMAGE_NAME" \
    flake8 src tests --count --exit-zero --max-complexity=25 --max-line-length=127 --statistics
FLAKE8_STATUS2=$?

# Запуск тестов (как в CI, но без coverage для скорости)
echo "🧪 Запуск тестов (pytest)..."
docker run --rm -v "$PROJECT_DIR:/app" -w /app "$IMAGE_NAME" \
    pytest tests/ -v --tb=short
PYTEST_STATUS=$?

# Проверяем результаты
if [ $BLACK_STATUS -ne 0 ]; then
    echo "❌ Ошибка форматирования кода. Запустите: docker run --rm -v \"\$PWD:/app\" $IMAGE_NAME black src tests"
    exit 1
fi

if [ $FLAKE8_STATUS -ne 0 ]; then
    echo "❌ Критические ошибки линтера обнаружены"
    exit 1
fi

if [ $FLAKE8_STATUS2 -ne 0 ]; then
    echo "❌ Ошибки линтера обнаружены"
    exit 1
fi

if [ $PYTEST_STATUS -ne 0 ]; then
    echo "❌ Тесты не прошли. Исправьте ошибки перед коммитом."
    exit 1
fi

echo "✅ Все проверки прошли успешно!"
exit 0
