"""Модуль для постобработки предсказаний модели и подготовки для API."""

import torch
import numpy as np
import logging
from typing import Union, List, Dict, Optional, Tuple
import pickle
import os


def postprocess_predictions(
    raw_predictions: Union[torch.Tensor, np.ndarray],
) -> np.ndarray:
    """
    Постобрабатывает сырые предсказания модели.

    Args:
        raw_predictions: Сырые предсказания модели (torch.Tensor или np.ndarray)

    Returns:
        np.ndarray: Обработанные предсказания в виде numpy массива
    """
    # Конвертируем в numpy если это torch.Tensor
    if isinstance(raw_predictions, torch.Tensor):
        predictions = raw_predictions.detach().cpu().numpy()
    else:
        predictions = np.array(raw_predictions)

    # Убираем лишние размерности
    predictions = predictions.squeeze()

    # Если остался 1D массив, делаем его 2D для единообразия
    if predictions.ndim == 1:
        predictions = predictions.reshape(-1, 1)

    return predictions


def clip_predictions(
    predictions: np.ndarray, min_value: float = 0.0, max_value: float = 5.0
) -> np.ndarray:
    """
    Ограничивает предсказания заданным диапазоном.

    Args:
        predictions: Предсказания модели
        min_value: Минимальное значение (по умолчанию 0.0)
        max_value: Максимальное значение (по умолчанию 5.0)

    Returns:
        np.ndarray: Ограниченные предсказания
    """
    return np.clip(predictions, min_value, max_value)


def format_predictions_for_api(
    predictions: np.ndarray, recipe_ids: Optional[List] = None
) -> List[Dict]:
    """
    Форматирует предсказания для API ответа.

    Args:
        predictions: Предсказания модели (1D массив)
        recipe_ids: Опциональный список ID рецептов

    Returns:
        List[Dict]: Список словарей с предсказаниями
    """
    predictions = predictions.squeeze()

    if predictions.ndim == 0:
        predictions = np.array([predictions])

    results = []
    for i, pred in enumerate(predictions):
        result = {"rating": float(pred), "rating_rounded": round(float(pred), 2)}

        if recipe_ids is not None and i < len(recipe_ids):
            result["recipe_id"] = recipe_ids[i]

        results.append(result)

    return results


class RecipePredictor:
    """Класс для инференса модели с постобработкой предсказаний."""

    def __init__(
        self,
        model,
        scaler_path: Optional[str] = None,
        vectorizer_path: Optional[str] = None,
    ):
        """
        Инициализирует RecipePredictor.

        Args:
            model: Обученная модель RecipeRankerModel
            scaler_path: Путь к сохраненному scaler (опционально)
            vectorizer_path: Путь к сохраненному vectorizer (опционально)
        """
        self.model = model
        self.model.eval()

        self.scaler = None
        self.vectorizer = None

        if scaler_path and os.path.exists(scaler_path):
            with open(scaler_path, "rb") as f:
                self.scaler = pickle.load(f)

        if vectorizer_path and os.path.exists(vectorizer_path):
            with open(vectorizer_path, "rb") as f:
                self.vectorizer = pickle.load(f)

    def predict(
        self,
        numerical_features: np.ndarray,
        text_features: Union[np.ndarray, List[str]],
        clip_values: bool = True,
        min_rating: float = 0.0,
        max_rating: float = 5.0,
    ) -> np.ndarray:
        """
        Делает предсказание для данных.

        Args:
            numerical_features: Числовые признаки (n_samples, n_features)
            text_features: Текстовые признаки (n_samples, n_features) или список строк
            clip_values: Ограничивать ли значения диапазоном [min_rating, max_rating]
            min_rating: Минимальное значение рейтинга
            max_rating: Максимальное значение рейтинга

        Returns:
            np.ndarray: Предсказания рейтингов
        """
        # Преобразуем в torch.Tensor
        if isinstance(numerical_features, np.ndarray):
            numerical_tensor = torch.tensor(numerical_features, dtype=torch.float32)
        else:
            numerical_tensor = numerical_features

        # Если text_features - список строк, векторизуем
        if isinstance(text_features, list) and self.vectorizer is not None:
            text_features = self.vectorizer.transform(text_features).toarray()

        if isinstance(text_features, np.ndarray):
            text_tensor = torch.tensor(text_features, dtype=torch.float32)
        else:
            text_tensor = text_features

        # Делаем предсказание
        with torch.no_grad():
            raw_predictions = self.model(
                numerical_features=numerical_tensor, text_features=text_tensor
            )

        # Постобработка
        predictions = postprocess_predictions(raw_predictions)

        # Ограничение значений
        if clip_values:
            predictions = clip_predictions(predictions, min_rating, max_rating)

        return predictions

    def predict_batch(
        self,
        numerical_features: np.ndarray,
        text_features: Union[np.ndarray, List[str]],
        recipe_ids: Optional[List] = None,
        clip_values: bool = True,
        min_rating: float = 0.0,
        max_rating: float = 5.0,
    ) -> List[Dict]:
        """
        Делает предсказания для батча и форматирует для API.

        Args:
            numerical_features: Числовые признаки
            text_features: Текстовые признаки или список строк
            recipe_ids: Опциональный список ID рецептов
            clip_values: Ограничивать ли значения
            min_rating: Минимальное значение рейтинга
            max_rating: Максимальное значение рейтинга

        Returns:
            List[Dict]: Список словарей с предсказаниями для API
        """
        predictions = self.predict(
            numerical_features, text_features, clip_values, min_rating, max_rating
        )

        return format_predictions_for_api(predictions, recipe_ids)


def validate_predictions(
    predictions: np.ndarray, min_value: float = 0.0, max_value: float = 5.0
) -> Tuple[bool, List[str]]:
    """
    Валидирует предсказания модели.

    Args:
        predictions: Предсказания модели
        min_value: Минимальное допустимое значение
        max_value: Максимальное допустимое значение

    Returns:
        Tuple[bool, List[str]]: (is_valid, список ошибок)
    """
    errors = []

    # Проверка типа
    if not isinstance(predictions, np.ndarray):
        errors.append("predictions должен быть np.ndarray")
        return False, errors

    # Проверка наличия NaN и Inf
    if np.isnan(predictions).any():
        errors.append("predictions содержит NaN")

    if np.isinf(predictions).any():
        errors.append("predictions содержит Inf")

    # Проверка диапазона значений
    if predictions.size > 0:
        pred_min = float(np.min(predictions))
        pred_max = float(np.max(predictions))

        if pred_min < min_value:
            errors.append(
                f"predictions содержит значения меньше {min_value}: минимальное = {pred_min}"
            )

        if pred_max > max_value:
            errors.append(
                f"predictions содержит значения больше {max_value}: максимальное = {pred_max}"
            )

    is_valid = len(errors) == 0
    if not is_valid:
        logging.warning(f"Обнаружены ошибки валидации предсказаний: {errors}")

    return is_valid, errors
