"""Тесты для модуля data_loader."""
import pytest
import pandas as pd
import numpy as np
from src.data_loader import (
    parse_duration_to_minutes,
    safe_list_eval_len,
    safe_join_list_from_str,
    count_steps,
    load_raw_data,
    calculate_review_statistics,
    merge_and_filter_data,
    extract_features,
    prepare_numerical_features,
    prepare_text_features,
    split_data,
    fit_scaler,
    transform_with_scaler,
    fit_vectorizer,
    transform_with_vectorizer
)


class TestParseDurationToMinutes:
    """Тесты для функции parse_duration_to_minutes."""
    
    def test_parse_hours_and_minutes(self):
        """Тест парсинга часов и минут."""
        assert parse_duration_to_minutes("PT2H30M") == 150
        assert parse_duration_to_minutes("PT1H15M") == 75
    
    def test_parse_only_hours(self):
        """Тест парсинга только часов."""
        assert parse_duration_to_minutes("PT3H") == 180
        assert parse_duration_to_minutes("PT1H") == 60
    
    def test_parse_only_minutes(self):
        """Тест парсинга только минут."""
        assert parse_duration_to_minutes("PT45M") == 45
        assert parse_duration_to_minutes("PT10M") == 10
    
    def test_parse_empty_string(self):
        """Тест парсинга пустой строки."""
        assert parse_duration_to_minutes("") == 0
        assert parse_duration_to_minutes("PT") == 0
    
    def test_parse_non_string(self):
        """Тест парсинга не-строки."""
        assert parse_duration_to_minutes(None) == 0
        assert parse_duration_to_minutes(123) == 0


class TestSafeListEvalLen:
    """Тесты для функции safe_list_eval_len."""
    
    def test_python_list_format(self):
        """Тест Python формата списка."""
        assert safe_list_eval_len("['item1', 'item2', 'item3']") == 3
        assert safe_list_eval_len("['apple']") == 1
    
    def test_r_format(self):
        """Тест R формата c()."""
        assert safe_list_eval_len('c("item1", "item2")') == 2
        assert safe_list_eval_len('c("apple", "banana", "cherry")') == 3
    
    def test_empty_list(self):
        """Тест пустого списка."""
        assert safe_list_eval_len("[]") == 0
        assert safe_list_eval_len('c()') == 0
    
    def test_invalid_format(self):
        """Тест невалидного формата."""
        assert safe_list_eval_len("not a list") == 0
        assert safe_list_eval_len("") == 0
        assert safe_list_eval_len(None) == 0


class TestSafeJoinListFromStr:
    """Тесты для функции safe_join_list_from_str."""
    
    def test_python_list_format(self):
        """Тест Python формата списка."""
        result = safe_join_list_from_str("['apple', 'banana']")
        assert result == "apple banana"
    
    def test_r_format(self):
        """Тест R формата c()."""
        result = safe_join_list_from_str('c("apple", "banana")')
        assert result == "apple banana"
    
    def test_empty_list(self):
        """Тест пустого списка."""
        assert safe_join_list_from_str("[]") == ""
        assert safe_join_list_from_str('c()') == ""
    
    def test_invalid_format(self):
        """Тест невалидного формата."""
        assert safe_join_list_from_str("not a list") == ""
        assert safe_join_list_from_str("") == ""
        assert safe_join_list_from_str(None) == ""


class TestCountSteps:
    """Тесты для функции count_steps."""
    
    def test_count_with_periods(self):
        """Тест подсчета шагов с точками."""
        instructions = "Step one. Step two. Step three."
        assert count_steps(instructions) == 3
    
    def test_count_with_exclamation(self):
        """Тест подсчета шагов с восклицательными знаками."""
        instructions = "Do this! Do that!"
        assert count_steps(instructions) == 2
    
    def test_count_empty_string(self):
        """Тест пустой строки."""
        # count_steps возвращает 1 для пустой строки (минимум один шаг)
        assert count_steps("") == 1
        # None обрабатывается как 0 в count_steps из-за pd.isna
        result = count_steps(None)
        assert result == 0  # count_steps возвращает 0 для None
    
    def test_count_single_step(self):
        """Тест одного шага."""
        instructions = "Just one step"
        assert count_steps(instructions) >= 1


class TestCalculateReviewStatistics:
    """Тесты для функции calculate_review_statistics."""
    
    def test_calculate_statistics(self):
        """Тест вычисления статистики."""
        reviews = pd.DataFrame({
            'RecipeId': [1, 1, 2, 2, 2],
            'Rating': [4, 5, 3, 4, 5]
        })
        
        result = calculate_review_statistics(reviews)
        
        assert 'RecipeId' in result.columns
        assert 'avg_rating' in result.columns
        assert 'review_count' in result.columns
        assert len(result) == 2
        assert result[result['RecipeId'] == 1]['avg_rating'].iloc[0] == 4.5
        assert result[result['RecipeId'] == 2]['review_count'].iloc[0] == 3


class TestPrepareNumericalFeatures:
    """Тесты для функции prepare_numerical_features."""
    
    def test_prepare_features(self):
        """Тест подготовки числовых признаков."""
        data = pd.DataFrame({
            'minutes': [30, 60, 90, 1000],  # 1000 - выброс
            'n_steps': [5, 10, 15, 20],
            'n_ingredients': [3, 5, 7, 9]
        })
        
        result = prepare_numerical_features(data)
        
        assert 'minutes' in result.columns
        assert 'n_steps' in result.columns
        assert 'n_ingredients' in result.columns
        # Проверка что выбросы обрезаны
        assert result['minutes'].max() <= data['minutes'].quantile(0.99)


class TestPrepareTextFeatures:
    """Тесты для функции prepare_text_features."""
    
    def test_prepare_text_features(self):
        """Тест подготовки текстовых признаков."""
        data = pd.DataFrame({
            'ingredients': ['apple banana', 'cherry', None, 'orange']
        })
        
        result = prepare_text_features(data)
        
        assert isinstance(result, pd.Series)
        assert len(result) == 4
        assert result.iloc[2] == ''  # None заменен на пустую строку


class TestFitScaler:
    """Тесты для функции fit_scaler."""
    
    def test_fit_scaler(self):
        """Тест обучения scaler."""
        X_train = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        
        scaler, X_scaled = fit_scaler(X_train)
        
        assert scaler is not None
        assert X_scaled.shape == X_train.shape
        # Проверка что данные нормализованы (среднее близко к 0)
        assert np.abs(X_scaled.mean()) < 1e-10


class TestTransformWithScaler:
    """Тесты для функции transform_with_scaler."""
    
    def test_transform_scaler(self):
        """Тест применения scaler."""
        from sklearn.preprocessing import StandardScaler
        
        X_train = np.array([[1.0, 2.0], [3.0, 4.0]])
        scaler = StandardScaler()
        scaler.fit(X_train)
        
        X_val = np.array([[2.0, 3.0]])
        X_transformed = transform_with_scaler(scaler, X_val)
        
        assert X_transformed.shape[0] == 1
        assert X_transformed.shape[1] == 2


class TestFitVectorizer:
    """Тесты для функции fit_vectorizer."""
    
    def test_fit_vectorizer(self):
        """Тест обучения vectorizer."""
        X_train_text = pd.Series(['apple banana', 'cherry', 'apple'])
        
        vectorizer, X_vec = fit_vectorizer(X_train_text, max_features=10)
        
        assert vectorizer is not None
        assert X_vec.shape[0] == 3
        assert X_vec.shape[1] <= 10


class TestSplitData:
    """Тесты для функции split_data."""
    
    def test_split_data(self):
        """Тест разделения данных."""
        numerical_features = pd.DataFrame({
            'minutes': [30, 60, 90, 120],
            'n_steps': [5, 10, 15, 20]
        })
        text_features = pd.Series(['text1', 'text2', 'text3', 'text4'])
        target = pd.Series([4.0, 4.5, 5.0, 3.5])
        
        X_train_num, X_val_num, X_train_text, X_val_text, y_train, y_val = split_data(
            numerical_features, text_features, target, test_size=0.25, random_state=42
        )
        
        assert len(X_train_num) == 3
        assert len(X_val_num) == 1
        assert len(X_train_text) == 3
        assert len(X_val_text) == 1
        assert len(y_train) == 3
        assert len(y_val) == 1

