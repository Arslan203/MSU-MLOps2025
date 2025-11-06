# Настройка автоматического запуска тестов при коммитах

Этот документ описывает, как настроить автоматический запуск тестов при каждом коммите в Git.

## Вариант 1: GitHub Actions (Рекомендуется)

### Уже настроено! ✅

GitHub Actions уже настроен и будет автоматически запускать тесты при каждом коммите.

**Файл конфигурации:** `.github/workflows/ci.yml`

**Что проверяется:**
- ✅ Линтер (flake8)
- ✅ Форматирование (black)
- ✅ Все тесты (pytest)
- ✅ Покрытие кода (coverage)

**Когда запускается:**
- При каждом `git push` в любую ветку
- При каждом Pull Request

**Проверить статус:**
1. Перейдите на GitHub в ваш репозиторий
2. Откройте вкладку "Actions"
3. Там будут видны все запуски CI

## Вариант 2: Pre-commit hooks (Локальная проверка)

Pre-commit hooks запускают проверки **перед** коммитом локально, чтобы не отправлять плохой код в репозиторий.

### Установка через pre-commit (Рекомендуется)

```bash
# 1. Установить pre-commit
pip install pre-commit

# 2. Активировать hooks
pre-commit install

# 3. (Опционально) Запустить на всех файлах один раз
pre-commit run --all-files
```

После установки при каждом `git commit` автоматически будут выполняться:
- Форматирование кода (black)
- Проверка линтера (flake8)
- Запуск тестов (pytest)

**Если проверки не пройдут, коммит будет отменен.**

### Установка простого bash hook

Альтернативный способ без pre-commit:

```bash
# 1. Скопировать hook
cp scripts/pre-commit-hook.sh .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit

# 2. Проверить что работает
.git/hooks/pre-commit
```

### Использование

**Нормальный коммит:**
```bash
git add .
git commit -m "добавлена новая функция"
# Автоматически запустятся проверки
```

**Пропустить проверки (не рекомендуется):**
```bash
git commit --no-verify -m "срочный коммит"
```

**Обновить hooks после изменения конфигурации:**
```bash
pre-commit install --overwrite
```

## Вариант 3: GitLab CI (если используете GitLab)

Если вы используете GitLab вместо GitHub, создайте файл `.gitlab-ci.yml`:

```yaml
stages:
  - test

test:
  stage: test
  image: python:3.9
  before_script:
    - pip install -r requirements.txt
  script:
    - flake8 src tests
    - black --check src tests
    - pytest tests/ -v --cov=src
  coverage: '/TOTAL.*\s+(\d+%)$/'
```

## Проверка работы

### GitHub Actions

1. Сделайте любой коммит:
```bash
git add .
git commit -m "test CI"
git push
```

2. Проверьте GitHub:
   - Откройте ваш репозиторий на GitHub
   - Перейдите в "Actions"
   - Должен появиться новый workflow run

### Pre-commit hooks

```bash
# Сделайте тестовый коммит
git add .
git commit -m "test pre-commit"
# Должны запуститься проверки
```

## Отключение проверок

### Отключить GitHub Actions

Удалите или переименуйте файл `.github/workflows/ci.yml`

### Отключить pre-commit

```bash
# Временно отключить
pre-commit uninstall

# Полностью удалить
rm .git/hooks/pre-commit
```

### Отключить на один коммит

```bash
git commit --no-verify -m "сообщение"
```

## Устранение проблем

### Pre-commit не запускается

```bash
# Проверить что установлен
ls -la .git/hooks/pre-commit

# Переустановить
pre-commit install --overwrite
```

### GitHub Actions не запускается

1. Проверьте что файл `.github/workflows/ci.yml` существует
2. Проверьте синтаксис YAML файла
3. Убедитесь что файл закоммичен в репозиторий
4. Проверьте вкладку "Actions" на GitHub

### Тесты падают в CI

1. Проверьте логи в GitHub Actions
2. Убедитесь что зависимости установлены правильно
3. Запустите тесты локально: `pytest tests/ -v`

## Рекомендации

1. **Всегда используйте GitHub Actions** для проверки в репозитории
2. **Используйте pre-commit hooks** для быстрой локальной проверки
3. **Не пропускайте проверки** без веской причины (--no-verify)
4. **Исправляйте ошибки** перед коммитом, а не после

