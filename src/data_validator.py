"""Модуль для валидации данных на разных этапах пайплайна."""
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional


def validate_raw_data(recipes: pd.DataFrame, reviews: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Валидирует сырые данные из CSV файлов.
    
    Args:
        recipes: DataFrame с рецептами
        reviews: DataFrame с отзывами
        
    Returns:
        Tuple[bool, List[str]]: (is_valid, список ошибок)
    """
    errors = []
    
    # Проверка наличия необходимых столбцов в recipes
    required_recipe_cols = ['RecipeId', 'TotalTime', 'RecipeInstructions', 'RecipeIngredientParts']
    missing_recipe_cols = [col for col in required_recipe_cols if col not in recipes.columns]
    if missing_recipe_cols:
        errors.append(f"Отсутствуют необходимые столбцы в recipes: {missing_recipe_cols}")
    
    # Проверка наличия необходимых столбцов в reviews
    required_review_cols = ['RecipeId', 'Rating']
    missing_review_cols = [col for col in required_review_cols if col not in reviews.columns]
    if missing_review_cols:
        errors.append(f"Отсутствуют необходимые столбцы в reviews: {missing_review_cols}")
    
    # Проверка типов данных
    if not recipes.empty:
        if 'RecipeId' in recipes.columns:
            if not pd.api.types.is_numeric_dtype(recipes['RecipeId']):
                errors.append("RecipeId в recipes должен быть числовым")
    
    if not reviews.empty:
        if 'RecipeId' in reviews.columns:
            if not pd.api.types.is_numeric_dtype(reviews['RecipeId']):
                errors.append("RecipeId в reviews должен быть числовым")
        if 'Rating' in reviews.columns:
            if not pd.api.types.is_numeric_dtype(reviews['Rating']):
                errors.append("Rating в reviews должен быть числовым")
            # Проверка диапазона рейтингов (обычно 0-5)
            ratings = reviews['Rating'].dropna()
            if not ratings.empty:
                if ratings.min() < 0 or ratings.max() > 5:
                    errors.append(f"Rating должен быть в диапазоне [0, 5], получен [{ratings.min()}, {ratings.max()}]")
    
    # Проверка наличия данных
    if recipes.empty:
        errors.append("DataFrame recipes пуст")
    if reviews.empty:
        errors.append("DataFrame reviews пуст")
    
    is_valid = len(errors) == 0
    if not is_valid:
        logging.warning(f"Обнаружены ошибки валидации сырых данных: {errors}")
    
    return is_valid, errors


def validate_merged_data(data: pd.DataFrame, min_reviews_per_recipe: int) -> Tuple[bool, List[str]]:
    """
    Валидирует данные после объединения и фильтрации.
    
    Args:
        data: DataFrame после объединения recipes и reviews
        min_reviews_per_recipe: Минимальное количество отзывов
        
    Returns:
        Tuple[bool, List[str]]: (is_valid, список ошибок)
    """
    errors = []
    
    # Проверка наличия необходимых столбцов
    required_cols = ['RecipeId', 'avg_rating', 'review_count', 'TotalTime', 
                     'RecipeInstructions', 'RecipeIngredientParts']
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        errors.append(f"Отсутствуют необходимые столбцы: {missing_cols}")
    
    # Проверка что данные не пусты
    if data.empty:
        errors.append("DataFrame data пуст после фильтрации")
        return False, errors
    
    # Проверка диапазона avg_rating
    if 'avg_rating' in data.columns:
        ratings = data['avg_rating'].dropna()
        if not ratings.empty:
            if ratings.min() < 0 or ratings.max() > 5:
                errors.append(f"avg_rating должен быть в диапазоне [0, 5], получен [{ratings.min()}, {ratings.max()}]")
    
    # Проверка что review_count >= min_reviews_per_recipe
    if 'review_count' in data.columns:
        invalid_counts = data[data['review_count'] < min_reviews_per_recipe]
        if not invalid_counts.empty:
            errors.append(f"Найдены рецепты с review_count < {min_reviews_per_recipe}")
    
    is_valid = len(errors) == 0
    if not is_valid:
        logging.warning(f"Обнаружены ошибки валидации объединенных данных: {errors}")
    
    return is_valid, errors


def validate_features(numerical_features: pd.DataFrame, text_features: pd.Series, 
                     target: pd.Series) -> Tuple[bool, List[str]]:
    """
    Валидирует извлеченные признаки.
    
    Args:
        numerical_features: DataFrame с числовыми признаками
        text_features: Series с текстовыми признаками
        target: Series с целевой переменной
        
    Returns:
        Tuple[bool, List[str]]: (is_valid, список ошибок)
    """
    errors = []
    
    # Проверка наличия необходимых числовых признаков
    required_num_cols = ['minutes', 'n_steps', 'n_ingredients']
    missing_num_cols = [col for col in required_num_cols if col not in numerical_features.columns]
    if missing_num_cols:
        errors.append(f"Отсутствуют необходимые числовые признаки: {missing_num_cols}")
    
    # Проверка типов данных
    for col in required_num_cols:
        if col in numerical_features.columns:
            if not pd.api.types.is_numeric_dtype(numerical_features[col]):
                errors.append(f"{col} должен быть числовым")
    
    # Проверка диапазонов значений
    if 'minutes' in numerical_features.columns:
        minutes = numerical_features['minutes'].dropna()
        if not minutes.empty:
            if minutes.min() < 0:
                errors.append(f"minutes не может быть отрицательным, минимальное значение: {minutes.min()}")
    
    if 'n_steps' in numerical_features.columns:
        n_steps = numerical_features['n_steps'].dropna()
        if not n_steps.empty:
            if n_steps.min() < 0:
                errors.append(f"n_steps не может быть отрицательным, минимальное значение: {n_steps.min()}")
    
    if 'n_ingredients' in numerical_features.columns:
        n_ingredients = numerical_features['n_ingredients'].dropna()
        if not n_ingredients.empty:
            if n_ingredients.min() < 0:
                errors.append(f"n_ingredients не может быть отрицательным, минимальное значение: {n_ingredients.min()}")
    
    # Проверка текстовых признаков
    if not isinstance(text_features, pd.Series):
        errors.append("text_features должен быть pd.Series")
    
    # Проверка целевой переменной
    if not isinstance(target, pd.Series):
        errors.append("target должен быть pd.Series")
    elif not target.empty:
        if not pd.api.types.is_numeric_dtype(target):
            errors.append("target должен быть числовым")
        else:
            target_values = target.dropna()
            if not target_values.empty:
                if target_values.min() < 0 or target_values.max() > 5:
                    errors.append(f"target должен быть в диапазоне [0, 5], получен [{target_values.min()}, {target_values.max()}]")
    
    # Проверка соответствия размеров
    if len(numerical_features) != len(text_features):
        errors.append(f"Размеры numerical_features ({len(numerical_features)}) и text_features ({len(text_features)}) не совпадают")
    
    if len(numerical_features) != len(target):
        errors.append(f"Размеры numerical_features ({len(numerical_features)}) и target ({len(target)}) не совпадают")
    
    is_valid = len(errors) == 0
    if not is_valid:
        logging.warning(f"Обнаружены ошибки валидации признаков: {errors}")
    
    return is_valid, errors


def validate_processed_features(X_train_num: np.ndarray, X_val_num: np.ndarray,
                                X_train_text: np.ndarray, X_val_text: np.ndarray,
                                y_train: pd.Series, y_val: pd.Series) -> Tuple[bool, List[str]]:
    """
    Валидирует обработанные признаки после нормализации и векторизации.
    
    Args:
        X_train_num: Массив числовых признаков для обучения
        X_val_num: Массив числовых признаков для валидации
        X_train_text: Массив текстовых признаков для обучения
        X_val_text: Массив текстовых признаков для валидации
        y_train: Целевая переменная для обучения
        y_val: Целевая переменная для валидации
        
    Returns:
        Tuple[bool, List[str]]: (is_valid, список ошибок)
    """
    errors = []
    
    # Проверка типов
    if not isinstance(X_train_num, np.ndarray):
        errors.append("X_train_num должен быть np.ndarray")
    if not isinstance(X_val_num, np.ndarray):
        errors.append("X_val_num должен быть np.ndarray")
    if not isinstance(X_train_text, np.ndarray):
        errors.append("X_train_text должен быть np.ndarray")
    if not isinstance(X_val_text, np.ndarray):
        errors.append("X_val_text должен быть np.ndarray")
    
    # Проверка размеров
    if len(X_train_num) != len(X_train_text):
        errors.append(f"Размеры X_train_num ({len(X_train_num)}) и X_train_text ({len(X_train_text)}) не совпадают")
    
    if len(X_val_num) != len(X_val_text):
        errors.append(f"Размеры X_val_num ({len(X_val_num)}) и X_val_text ({len(X_val_text)}) не совпадают")
    
    if len(X_train_num) != len(y_train):
        errors.append(f"Размеры X_train_num ({len(X_train_num)}) и y_train ({len(y_train)}) не совпадают")
    
    if len(X_val_num) != len(y_val):
        errors.append(f"Размеры X_val_num ({len(X_val_num)}) и y_val ({len(y_val)}) не совпадают")
    
    # Проверка что данные не пусты
    if X_train_num.size == 0:
        errors.append("X_train_num пуст")
    if X_val_num.size == 0:
        errors.append("X_val_num пуст")
    if X_train_text.size == 0:
        errors.append("X_train_text пуст")
    if X_val_text.size == 0:
        errors.append("X_val_text пуст")
    
    # Проверка наличия NaN и Inf
    if np.isnan(X_train_num).any():
        errors.append("X_train_num содержит NaN")
    if np.isnan(X_val_num).any():
        errors.append("X_val_num содержит NaN")
    if np.isnan(X_train_text).any():
        errors.append("X_train_text содержит NaN")
    if np.isnan(X_val_text).any():
        errors.append("X_val_text содержит NaN")
    
    if np.isinf(X_train_num).any():
        errors.append("X_train_num содержит Inf")
    if np.isinf(X_val_num).any():
        errors.append("X_val_num содержит Inf")
    if np.isinf(X_train_text).any():
        errors.append("X_train_text содержит Inf")
    if np.isinf(X_val_text).any():
        errors.append("X_val_text содержит Inf")
    
    is_valid = len(errors) == 0
    if not is_valid:
        logging.warning(f"Обнаружены ошибки валидации обработанных признаков: {errors}")
    
    return is_valid, errors


def get_data_statistics(data: pd.DataFrame) -> Dict:
    """
    Вычисляет базовую статистику для оценки корректности данных.
    
    Args:
        data: DataFrame для анализа
        
    Returns:
        Dict: Словарь со статистикой
    """
    stats = {
        'total_recipes': len(data),
        'columns': list(data.columns),
        'missing_values': data.isnull().sum().to_dict(),
        'dtypes': data.dtypes.to_dict()
    }
    
    # Статистика для числовых столбцов
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        stats['numeric_statistics'] = data[numeric_cols].describe().to_dict()
    
    # Статистика для avg_rating если есть
    if 'avg_rating' in data.columns:
        stats['rating_statistics'] = {
            'mean': float(data['avg_rating'].mean()),
            'std': float(data['avg_rating'].std()),
            'min': float(data['avg_rating'].min()),
            'max': float(data['avg_rating'].max()),
            'median': float(data['avg_rating'].median())
        }
    
    return stats

