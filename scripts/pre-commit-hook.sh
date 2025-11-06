#!/bin/bash
# Pre-commit hook для запуска тестов перед коммитом
# Точная копия проверок из GitHub Actions CI
# Установка: cp scripts/pre-commit-hook.sh .git/hooks/pre-commit && chmod +x .git/hooks/pre-commit

echo "🔍 Запуск проверок перед коммитом (как в CI)..."

# Определяем Python команду (python3 или python)
if command -v python3 &> /dev/null; then
    PYTHON=python3
elif command -v python &> /dev/null; then
    PYTHON=python
else
    echo "❌ Python не найден. Пропускаем локальные проверки."
    echo "💡 Проверки будут выполнены в GitHub Actions CI."
    exit 0
fi

# Проверка форматирования с black (как в CI)
echo "📝 Проверка форматирования кода (black)..."
if command -v black &> /dev/null; then
    black --check src tests
elif $PYTHON -m black --version &> /dev/null 2>&1; then
    $PYTHON -m black --check src tests
else
    echo "⚠️  black не найден. Пропускаем проверку форматирования."
    echo "💡 Проверка будет выполнена в GitHub Actions CI."
fi
BLACK_STATUS=$?

# Проверка линтера - первая часть (как в CI)
echo "🔎 Проверка линтера flake8 (критические ошибки)..."
FLAKE8_STATUS=0
if command -v flake8 &> /dev/null; then
    flake8 src tests --count --select=E9,F63,F7,F82 --show-source --statistics
    FLAKE8_STATUS=$?
elif $PYTHON -m flake8 --version &> /dev/null 2>&1; then
    $PYTHON -m flake8 src tests --count --select=E9,F63,F7,F82 --show-source --statistics
    FLAKE8_STATUS=$?
else
    echo "⚠️  flake8 не найден. Пропускаем проверку линтера."
    echo "💡 Проверка будет выполнена в GitHub Actions CI."
fi

# Проверка линтера - вторая часть (как в CI)
echo "🔎 Проверка линтера flake8 (полная проверка)..."
FLAKE8_STATUS2=0
if command -v flake8 &> /dev/null; then
    flake8 src tests --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
    FLAKE8_STATUS2=$?
elif $PYTHON -m flake8 --version &> /dev/null 2>&1; then
    $PYTHON -m flake8 src tests --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
    FLAKE8_STATUS2=$?
else
    echo "⚠️  flake8 не найден. Пропускаем проверку линтера."
    echo "💡 Проверка будет выполнена в GitHub Actions CI."
fi

# Запуск тестов (как в CI, но без coverage для скорости)
echo "🧪 Запуск тестов (pytest)..."
PYTEST_STATUS=0
if command -v pytest &> /dev/null; then
    pytest tests/ -v --tb=short
    PYTEST_STATUS=$?
elif $PYTHON -m pytest --version &> /dev/null 2>&1; then
    $PYTHON -m pytest tests/ -v --tb=short
    PYTEST_STATUS=$?
else
    echo "⚠️  pytest не найден. Пропускаем тесты."
    echo "💡 Тесты будут выполнены в GitHub Actions CI."
fi

# Проверяем результаты
if [ $BLACK_STATUS -ne 0 ]; then
    echo "❌ Ошибка форматирования кода. Запустите: $PYTHON -m black src tests"
    exit 1
fi

if [ $FLAKE8_STATUS -ne 0 ] || [ $FLAKE8_STATUS2 -ne 0 ]; then
    echo "❌ Ошибки линтера обнаружены"
    exit 1
fi

if [ $PYTEST_STATUS -ne 0 ]; then
    echo "❌ Тесты не прошли. Исправьте ошибки перед коммитом."
    exit 1
fi

echo "✅ Все проверки прошли успешно!"
exit 0

