"""Тесты для модуля data_validator."""

import pandas as pd
import numpy as np
from src.data_validator import (
    validate_raw_data,
    validate_merged_data,
    validate_features,
    validate_processed_features,
    get_data_statistics,
)


class TestValidateRawData:
    """Тесты для функции validate_raw_data."""

    def test_valid_data(self):
        """Тест валидных данных."""
        recipes = pd.DataFrame(
            {
                "RecipeId": [1, 2, 3],
                "TotalTime": ["PT30M", "PT60M", "PT90M"],
                "RecipeInstructions": [
                    "Step 1. Step 2.",
                    "Step 1.",
                    "Step 1. Step 2. Step 3.",
                ],
                "RecipeIngredientParts": [
                    '["apple", "banana"]',
                    '["cherry"]',
                    '["orange"]',
                ],
            }
        )
        reviews = pd.DataFrame({"RecipeId": [1, 1, 2], "Rating": [4.0, 5.0, 3.5]})

        is_valid, errors = validate_raw_data(recipes, reviews)
        assert is_valid
        assert len(errors) == 0

    def test_missing_columns_recipes(self):
        """Тест отсутствующих столбцов в recipes."""
        recipes = pd.DataFrame({"RecipeId": [1, 2]})
        reviews = pd.DataFrame({"RecipeId": [1], "Rating": [4.0]})

        is_valid, errors = validate_raw_data(recipes, reviews)
        assert not is_valid
        assert any("recipes" in err.lower() for err in errors)

    def test_missing_columns_reviews(self):
        """Тест отсутствующих столбцов в reviews."""
        recipes = pd.DataFrame(
            {
                "RecipeId": [1],
                "TotalTime": ["PT30M"],
                "RecipeInstructions": ["Step 1."],
                "RecipeIngredientParts": ['["apple"]'],
            }
        )
        reviews = pd.DataFrame({"RecipeId": [1]})

        is_valid, errors = validate_raw_data(recipes, reviews)
        assert not is_valid
        assert any("reviews" in err.lower() for err in errors)

    def test_invalid_rating_range(self):
        """Тест невалидного диапазона рейтингов."""
        recipes = pd.DataFrame(
            {
                "RecipeId": [1],
                "TotalTime": ["PT30M"],
                "RecipeInstructions": ["Step 1."],
                "RecipeIngredientParts": ['["apple"]'],
            }
        )
        reviews = pd.DataFrame({"RecipeId": [1], "Rating": [6.0]})  # Невалидный рейтинг

        is_valid, errors = validate_raw_data(recipes, reviews)
        assert not is_valid
        assert any("rating" in err.lower() for err in errors)

    def test_empty_dataframes(self):
        """Тест пустых DataFrame."""
        recipes = pd.DataFrame()
        reviews = pd.DataFrame()

        is_valid, errors = validate_raw_data(recipes, reviews)
        assert not is_valid
        assert any("пуст" in err.lower() for err in errors)


class TestValidateMergedData:
    """Тесты для функции validate_merged_data."""

    def test_valid_merged_data(self):
        """Тест валидных объединенных данных."""
        data = pd.DataFrame(
            {
                "RecipeId": [1, 2],
                "avg_rating": [4.5, 3.5],
                "review_count": [10, 15],
                "TotalTime": ["PT30M", "PT60M"],
                "RecipeInstructions": ["Step 1.", "Step 1. Step 2."],
                "RecipeIngredientParts": ['["apple"]', '["banana"]'],
            }
        )

        is_valid, errors = validate_merged_data(data, min_reviews_per_recipe=5)
        assert is_valid
        assert len(errors) == 0

    def test_missing_columns(self):
        """Тест отсутствующих столбцов."""
        data = pd.DataFrame({"RecipeId": [1]})

        is_valid, errors = validate_merged_data(data, min_reviews_per_recipe=5)
        assert not is_valid
        assert len(errors) > 0

    def test_invalid_review_count(self):
        """Тест невалидного количества отзывов."""
        data = pd.DataFrame(
            {
                "RecipeId": [1],
                "avg_rating": [4.5],
                "review_count": [3],  # Меньше минимума
                "TotalTime": ["PT30M"],
                "RecipeInstructions": ["Step 1."],
                "RecipeIngredientParts": ['["apple"]'],
            }
        )

        is_valid, errors = validate_merged_data(data, min_reviews_per_recipe=5)
        assert not is_valid


class TestValidateFeatures:
    """Тесты для функции validate_features."""

    def test_valid_features(self):
        """Тест валидных признаков."""
        numerical_features = pd.DataFrame(
            {"minutes": [30, 60], "n_steps": [5, 10], "n_ingredients": [3, 5]}
        )
        text_features = pd.Series(["apple banana", "cherry"])
        target = pd.Series([4.5, 3.5])

        is_valid, errors = validate_features(numerical_features, text_features, target)
        assert is_valid
        assert len(errors) == 0

    def test_missing_numerical_features(self):
        """Тест отсутствующих числовых признаков."""
        numerical_features = pd.DataFrame({"minutes": [30]})
        text_features = pd.Series(["apple"])
        target = pd.Series([4.5])

        is_valid, errors = validate_features(numerical_features, text_features, target)
        assert not is_valid

    def test_invalid_target_range(self):
        """Тест невалидного диапазона целевой переменной."""
        numerical_features = pd.DataFrame(
            {"minutes": [30], "n_steps": [5], "n_ingredients": [3]}
        )
        text_features = pd.Series(["apple"])
        target = pd.Series([6.0])  # Невалидный рейтинг

        is_valid, errors = validate_features(numerical_features, text_features, target)
        assert not is_valid

    def test_mismatched_lengths(self):
        """Тест несоответствия размеров."""
        numerical_features = pd.DataFrame(
            {"minutes": [30, 60], "n_steps": [5, 10], "n_ingredients": [3, 5]}
        )
        text_features = pd.Series(["apple"])  # Неправильный размер
        target = pd.Series([4.5, 3.5])

        is_valid, errors = validate_features(numerical_features, text_features, target)
        assert not is_valid


class TestValidateProcessedFeatures:
    """Тесты для функции validate_processed_features."""

    def test_valid_processed_features(self):
        """Тест валидных обработанных признаков."""
        X_train_num = np.array([[1.0, 2.0], [3.0, 4.0]])
        X_val_num = np.array([[2.0, 3.0]])
        X_train_text = np.array([[0.5, 0.3], [0.2, 0.8]])
        X_val_text = np.array([[0.4, 0.6]])
        y_train = pd.Series([4.5, 3.5])
        y_val = pd.Series([4.0])

        is_valid, errors = validate_processed_features(
            X_train_num, X_val_num, X_train_text, X_val_text, y_train, y_val
        )
        assert is_valid
        assert len(errors) == 0

    def test_nan_values(self):
        """Тест NaN значений."""
        X_train_num = np.array([[1.0, np.nan], [3.0, 4.0]])
        X_val_num = np.array([[2.0, 3.0]])
        X_train_text = np.array([[0.5, 0.3], [0.2, 0.8]])
        X_val_text = np.array([[0.4, 0.6]])
        y_train = pd.Series([4.5, 3.5])
        y_val = pd.Series([4.0])

        is_valid, errors = validate_processed_features(
            X_train_num, X_val_num, X_train_text, X_val_text, y_train, y_val
        )
        assert not is_valid
        assert any("nan" in err.lower() for err in errors)

    def test_mismatched_lengths(self):
        """Тест несоответствия размеров."""
        X_train_num = np.array([[1.0, 2.0], [3.0, 4.0]])
        X_val_num = np.array([[2.0, 3.0]])
        X_train_text = np.array([[0.5, 0.3]])  # Неправильный размер
        X_val_text = np.array([[0.4, 0.6]])
        y_train = pd.Series([4.5, 3.5])
        y_val = pd.Series([4.0])

        is_valid, errors = validate_processed_features(
            X_train_num, X_val_num, X_train_text, X_val_text, y_train, y_val
        )
        assert not is_valid


class TestGetDataStatistics:
    """Тесты для функции get_data_statistics."""

    def test_get_statistics(self):
        """Тест получения статистики."""
        data = pd.DataFrame(
            {
                "RecipeId": [1, 2, 3],
                "avg_rating": [4.5, 3.5, 5.0],
                "minutes": [30, 60, 90],
            }
        )

        stats = get_data_statistics(data)

        assert "total_recipes" in stats
        assert stats["total_recipes"] == 3
        assert "columns" in stats
        assert "missing_values" in stats
        assert "numeric_statistics" in stats
