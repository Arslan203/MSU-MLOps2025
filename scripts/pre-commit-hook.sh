#!/bin/bash
# Pre-commit hook для запуска тестов перед коммитом
# Устанавливается автоматически при выполнении: pre-commit install

echo "🔍 Запуск проверок перед коммитом..."

# Проверка форматирования с black
echo "📝 Проверка форматирования кода (black)..."
black --check src tests
if [ $? -ne 0 ]; then
    echo "❌ Ошибка форматирования кода. Запустите: black src tests"
    exit 1
fi

# Проверка линтера
echo "🔎 Проверка линтера (flake8)..."
flake8 src tests --max-line-length=127 --max-complexity=10 --count
if [ $? -ne 0 ]; then
    echo "❌ Ошибки линтера обнаружены"
    exit 1
fi

# Запуск тестов
echo "🧪 Запуск тестов (pytest)..."
pytest tests/ -v --tb=short
if [ $? -ne 0 ]; then
    echo "❌ Тесты не прошли. Исправьте ошибки перед коммитом."
    exit 1
fi

echo "✅ Все проверки прошли успешно!"
exit 0

