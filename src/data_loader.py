import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
from torch.utils.data import Dataset, DataLoader
import logging
import pickle
import os
import ast
import re

from .data_validator import (
    validate_raw_data,
    validate_merged_data,
    validate_features,
    validate_processed_features,
    get_data_statistics
)

def parse_duration_to_minutes(duration_str):
    if not isinstance(duration_str, str):
        return 0
    hours = 0
    minutes = 0
    try:
        hours_match = re.search(r'(\d+)H', duration_str)
        if hours_match:
            hours = int(hours_match.group(1))
        minutes_match = re.search(r'(\d+)M', duration_str)
        if minutes_match:
            minutes = int(minutes_match.group(1))
    except (TypeError, AttributeError):
        return 0
    return hours * 60 + minutes

def safe_list_eval_len(s):
    if isinstance(s, str):
        try:
            # Пробуем Python формат ['item1', 'item2']
            result = ast.literal_eval(s)
            if isinstance(result, list):
                return len(result)
        except (ValueError, SyntaxError):
            pass
        # Пробуем R формат c("item1", "item2")
        try:
            # Убираем c( и закрывающую скобку
            if s.startswith('c(') and s.endswith(')'):
                inner = s[2:-1]  # Убираем c( и )
                # Парсим как Python список
                result = ast.literal_eval(inner)
                if isinstance(result, tuple):
                    result = list(result)
                if isinstance(result, list):
                    return len(result)
        except (ValueError, SyntaxError):
            pass
    return 0

def safe_join_list_from_str(s):
    if isinstance(s, str):
        try:
            # Пробуем Python формат ['item1', 'item2']
            result = ast.literal_eval(s)
            if isinstance(result, list):
                return ' '.join(str(item) for item in result)
        except (ValueError, SyntaxError):
            pass
        # Пробуем R формат c("item1", "item2")
        try:
            # Убираем c( и закрывающую скобку
            if s.startswith('c(') and s.endswith(')'):
                inner = s[2:-1]  # Убираем c( и )
                # Парсим как Python кортеж/список
                result = ast.literal_eval(inner)
                if isinstance(result, tuple):
                    result = list(result)
                if isinstance(result, list):
                    return ' '.join(str(item) for item in result)
        except (ValueError, SyntaxError):
            pass
    return ""


def count_steps(instructions):
    """Подсчитывает количество шагов в инструкциях по приготовлению."""
    if pd.isna(instructions) or not isinstance(instructions, str):
        return 0
    # Разделяем по точкам, восклицательным и вопросительным знакам
    steps = re.split(r'[.!?]+', instructions)
    # Убираем пустые строки
    steps = [s.strip() for s in steps if s.strip()]
    return len(steps) if len(steps) > 0 else 1


def load_raw_data(recipes_path, reviews_path):
    """Загружает сырые данные из CSV файлов."""
    logging.info("Загрузка данных...")
    recipes = pd.read_csv(recipes_path)
    reviews = pd.read_csv(reviews_path)
    
    # Валидация сырых данных
    is_valid, errors = validate_raw_data(recipes, reviews)
    if not is_valid:
        raise ValueError(f"Ошибки валидации сырых данных: {errors}")
    
    # Логируем статистику
    stats = get_data_statistics(recipes)
    logging.info(f"Статистика recipes: {stats['total_recipes']} рецептов, {len(stats['columns'])} столбцов")
    
    return recipes, reviews


def calculate_review_statistics(reviews):
    """Вычисляет средний рейтинг и количество отзывов для каждого рецепта."""
    review_summary = reviews.groupby('RecipeId').agg(
        avg_rating=('Rating', 'mean'),
        review_count=('Rating', 'count')
    ).reset_index()
    return review_summary


def merge_and_filter_data(recipes, review_summary, min_reviews_per_recipe):
    """Объединяет рецепты с отзывами и фильтрует по минимальному количеству отзывов."""
    data = pd.merge(recipes, review_summary, left_on='RecipeId', right_on='RecipeId')
    data = data[data['review_count'] >= min_reviews_per_recipe]
    logging.info(f"Размер датасета после фильтрации: {data.shape[0]} рецептов")
    
    # Валидация объединенных данных
    is_valid, errors = validate_merged_data(data, min_reviews_per_recipe)
    if not is_valid:
        raise ValueError(f"Ошибки валидации объединенных данных: {errors}")
    
    return data


def extract_features(data):
    """Извлекает и подготавливает признаки из данных."""
    # Парсим время приготовления из TotalTime (ISO 8601 формат PT24H45M)
    data['minutes'] = data['TotalTime'].apply(parse_duration_to_minutes)
    
    # Извлекаем количество шагов из RecipeInstructions
    data['n_steps'] = data['RecipeInstructions'].apply(count_steps)
    
    # Извлекаем количество и список ингредиентов из RecipeIngredientParts
    data['n_ingredients'] = data['RecipeIngredientParts'].apply(safe_list_eval_len)
    data['ingredients'] = data['RecipeIngredientParts'].apply(safe_join_list_from_str)
    
    return data


def prepare_numerical_features(data):
    """Подготавливает числовые признаки с обработкой выбросов."""
    numerical_features = data[['minutes', 'n_steps', 'n_ingredients']].fillna(0)
    
    # Ограничиваем выбросы (clip)
    upper_bounds = numerical_features.quantile(0.99)
    numerical_features = numerical_features.clip(lower=0, upper=upper_bounds, axis=1)
    
    return numerical_features


def prepare_text_features(data):
    """Подготавливает текстовые признаки (ингредиенты)."""
    text_features = data['ingredients'].fillna('')
    return text_features


def split_data(numerical_features, text_features, target, test_size, random_state):
    """Разделяет данные на обучающую и валидационную выборки."""
    X_train_num, X_val_num, X_train_text, X_val_text, y_train, y_val = train_test_split(
        numerical_features, text_features, target,
        test_size=test_size,
        random_state=random_state
    )
    return X_train_num, X_val_num, X_train_text, X_val_text, y_train, y_val


def fit_scaler(X_train_num):
    """Обучает StandardScaler на обучающих данных."""
    scaler = StandardScaler()
    X_train_num_scaled = scaler.fit_transform(X_train_num)
    return scaler, X_train_num_scaled


def transform_with_scaler(scaler, X_num):
    """Применяет обученный scaler к данным."""
    return scaler.transform(X_num)


def fit_vectorizer(X_train_text, max_features):
    """Обучает TfidfVectorizer на обучающих данных."""
    vectorizer = TfidfVectorizer(max_features=max_features)
    X_train_text_vec = vectorizer.fit_transform(X_train_text).toarray()
    return vectorizer, X_train_text_vec


def transform_with_vectorizer(vectorizer, X_text):
    """Применяет обученный vectorizer к данным."""
    return vectorizer.transform(X_text).toarray()


def save_artifacts(scaler, vectorizer, artifacts_path):
    """Сохраняет артефакты предобработки (scaler, vectorizer)."""
    os.makedirs(artifacts_path, exist_ok=True)
    with open(os.path.join(artifacts_path, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    with open(os.path.join(artifacts_path, 'vectorizer.pkl'), 'wb') as f:
        pickle.dump(vectorizer, f)
    logging.info(f"Артефакты (scaler, vectorizer) сохранены в {artifacts_path}")


def create_dataloaders(X_train_num, X_train_text, y_train, X_val_num, X_val_text, y_val, batch_size):
    """Создает DataLoader'ы для обучения и валидации."""
    train_dataset = RecipeDataset(X_train_num, X_train_text, y_train.values)
    val_dataset = RecipeDataset(X_val_num, X_val_text, y_val.values)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def load_and_prepare_data(config):
    """Главная функция для загрузки и подготовки данных."""
    # 1. Загрузка данных
    recipes, reviews = load_raw_data(
        config['data']['recipes_path'],
        config['data']['reviews_path']
    )

    # 2. Расчет статистики по отзывам
    review_summary = calculate_review_statistics(reviews)

    # 3. Объединение и фильтрация
    data = merge_and_filter_data(
        recipes,
        review_summary,
        config['data']['min_reviews_per_recipe']
    )

    # 4. Извлечение признаков
    data = extract_features(data)

    # 5. Подготовка числовых и текстовых признаков
    numerical_features = prepare_numerical_features(data)
    text_features = prepare_text_features(data)
    target = data['avg_rating']
    
    # Валидация признаков
    is_valid, errors = validate_features(numerical_features, text_features, target)
    if not is_valid:
        raise ValueError(f"Ошибки валидации признаков: {errors}")

    # 6. Разделение на train/validation
    X_train_num, X_val_num, X_train_text, X_val_text, y_train, y_val = split_data(
        numerical_features,
        text_features,
        target,
        config['training']['test_size'],
        config['training']['random_seed']
    )

    # 7. Нормализация числовых признаков
    scaler, X_train_num_scaled = fit_scaler(X_train_num)
    X_val_num_scaled = transform_with_scaler(scaler, X_val_num)

    # 8. Векторизация текстовых признаков
    vectorizer, X_train_text_vec = fit_vectorizer(
        X_train_text,
        config['text_vectorizer']['max_features']
    )
    X_val_text_vec = transform_with_vectorizer(vectorizer, X_val_text)
    
    config['model']['vocab_size'] = len(vectorizer.vocabulary_)
    
    # Валидация обработанных признаков
    is_valid, errors = validate_processed_features(
        X_train_num_scaled, X_val_num_scaled,
        X_train_text_vec, X_val_text_vec,
        y_train, y_val
    )
    if not is_valid:
        raise ValueError(f"Ошибки валидации обработанных признаков: {errors}")

    # 9. Сохранение артефактов
    save_artifacts(
        scaler,
        vectorizer,
        config['data']['output_artifacts_path']
    )

    # 10. Создание DataLoader'ов
    train_loader, val_loader = create_dataloaders(
        X_train_num_scaled,
        X_train_text_vec,
        y_train,
        X_val_num_scaled,
        X_val_text_vec,
        y_val,
        config['training']['batch_size']
    )

    return train_loader, val_loader

class RecipeDataset(Dataset):
    def __init__(self, numerical_features, text_features, targets):
        self.numerical_features = torch.tensor(numerical_features, dtype=torch.float32)
        self.text_features = torch.tensor(text_features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return {
            "numerical_features": self.numerical_features[idx],
            "text_features": self.text_features[idx],
            "target": self.targets[idx].unsqueeze(0)
        }

