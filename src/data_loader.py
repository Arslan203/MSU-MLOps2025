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

def load_and_prepare_data(config):
    logging.info("Загрузка данных...")
    recipes = pd.read_csv(config['data']['recipes_path'])
    reviews = pd.read_csv(config['data']['reviews_path'])

    # 1. Расчет среднего рейтинга и фильтрация
    # Используем правильные имена столбцов из CSV (RecipeId, Rating)
    review_summary = reviews.groupby('RecipeId').agg(
        avg_rating=('Rating', 'mean'),
        review_count=('Rating', 'count')
    ).reset_index()

    data = pd.merge(recipes, review_summary, left_on='RecipeId', right_on='RecipeId')
    
    data = data[data['review_count'] >= config['data']['min_reviews_per_recipe']]
    logging.info(f"Размер датасета после фильтрации: {data.shape[0]} рецептов")
    
    # 2. Извлечение и подготовка признаков из реальных столбцов CSV
    # Парсим время приготовления из TotalTime (ISO 8601 формат PT24H45M)
    data['minutes'] = data['TotalTime'].apply(parse_duration_to_minutes)
    
    # Извлекаем количество шагов из RecipeInstructions (считаем предложения)
    def count_steps(instructions):
        if pd.isna(instructions) or not isinstance(instructions, str):
            return 0
        # Разделяем по точкам, восклицательным и вопросительным знакам
        steps = re.split(r'[.!?]+', instructions)
        # Убираем пустые строки
        steps = [s.strip() for s in steps if s.strip()]
        return len(steps) if len(steps) > 0 else 1
    
    data['n_steps'] = data['RecipeInstructions'].apply(count_steps)
    
    # Извлекаем количество и список ингредиентов из RecipeIngredientParts
    data['n_ingredients'] = data['RecipeIngredientParts'].apply(safe_list_eval_len)
    data['ingredients'] = data['RecipeIngredientParts'].apply(safe_join_list_from_str)
    
    # Используем подготовленные признаки
    numerical_features = data[['minutes', 'n_steps', 'n_ingredients']].fillna(0)
    
    # Ограничиваем выбросы (clip)
    upper_bounds = numerical_features.quantile(0.99)
    numerical_features = numerical_features.clip(lower=0, upper=upper_bounds, axis=1)
    
    # Текстовые признаки (ингредиенты) уже в виде строки
    text_features = data['ingredients'].fillna('')
    
    target = data['avg_rating']

    # 3. Разделение на train/validation
    X_train_num, X_val_num, X_train_text, X_val_text, y_train, y_val = train_test_split(
        numerical_features, text_features, target,
        test_size=config['training']['test_size'],
        random_state=config['training']['random_seed']
    )

    # 4. Нормализация числовых признаков
    scaler = StandardScaler()
    X_train_num_scaled = scaler.fit_transform(X_train_num)
    X_val_num_scaled = scaler.transform(X_val_num)

    # 5. Векторизация текстовых признаков (ингредиентов)
    vectorizer = TfidfVectorizer(max_features=config['text_vectorizer']['max_features'])
    X_train_text_vec = vectorizer.fit_transform(X_train_text).toarray()
    X_val_text_vec = vectorizer.transform(X_val_text).toarray()
    
    config['model']['vocab_size'] = len(vectorizer.vocabulary_)

    # 6. Сохранение артефактов
    artifacts_path = config['data']['output_artifacts_path']
    os.makedirs(artifacts_path, exist_ok=True)
    with open(os.path.join(artifacts_path, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    with open(os.path.join(artifacts_path, 'vectorizer.pkl'), 'wb') as f:
        pickle.dump(vectorizer, f)
    logging.info(f"Артефакты (scaler, vectorizer) сохранены в {artifacts_path}")

    # 7. Создание DataLoader'ов
    train_dataset = RecipeDataset(X_train_num_scaled, X_train_text_vec, y_train.values)
    val_dataset = RecipeDataset(X_val_num_scaled, X_val_text_vec, y_val.values)

    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False)

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

