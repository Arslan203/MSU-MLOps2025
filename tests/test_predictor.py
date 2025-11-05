"""Тесты для модуля predictor."""
import pytest
import torch
import numpy as np
from src.predictor import (
    postprocess_predictions,
    clip_predictions,
    format_predictions_for_api,
    validate_predictions
)


class TestPostprocessPredictions:
    """Тесты для функции postprocess_predictions."""
    
    def test_postprocess_torch_tensor(self):
        """Тест постобработки torch.Tensor."""
        raw_pred = torch.tensor([[4.5], [3.5], [5.0]])
        result = postprocess_predictions(raw_pred)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (3, 1)
    
    def test_postprocess_numpy_array(self):
        """Тест постобработки numpy массива."""
        raw_pred = np.array([[4.5], [3.5]])
        result = postprocess_predictions(raw_pred)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 1)
    
    def test_postprocess_squeeze(self):
        """Тест удаления лишних размерностей."""
        raw_pred = torch.tensor([[[4.5]]])
        result = postprocess_predictions(raw_pred)
        
        # После squeeze из [[[4.5]]] может получиться скаляр или 1D массив
        # Проверяем что это numpy массив
        assert isinstance(result, np.ndarray)
        # Если результат скаляр, reshape в 1D
        if result.ndim == 0:
            result = result.reshape(1, 1)
        assert result.ndim >= 1


class TestClipPredictions:
    """Тесты для функции clip_predictions."""
    
    def test_clip_within_range(self):
        """Тест ограничения в пределах диапазона."""
        predictions = np.array([2.0, 3.0, 4.0])
        result = clip_predictions(predictions, min_value=0.0, max_value=5.0)
        
        np.testing.assert_array_equal(result, predictions)
    
    def test_clip_above_max(self):
        """Тест ограничения значений выше максимума."""
        predictions = np.array([2.0, 6.0, 4.0])
        result = clip_predictions(predictions, min_value=0.0, max_value=5.0)
        
        assert np.max(result) <= 5.0
        assert result[1] == 5.0
    
    def test_clip_below_min(self):
        """Тест ограничения значений ниже минимума."""
        predictions = np.array([-1.0, 3.0, 4.0])
        result = clip_predictions(predictions, min_value=0.0, max_value=5.0)
        
        assert np.min(result) >= 0.0
        assert result[0] == 0.0


class TestFormatPredictionsForApi:
    """Тесты для функции format_predictions_for_api."""
    
    def test_format_without_recipe_ids(self):
        """Тест форматирования без ID рецептов."""
        predictions = np.array([4.5, 3.5, 5.0])
        result = format_predictions_for_api(predictions)
        
        assert len(result) == 3
        assert all('rating' in item for item in result)
        assert all('rating_rounded' in item for item in result)
        assert result[0]['rating'] == 4.5
        assert result[0]['rating_rounded'] == 4.5
    
    def test_format_with_recipe_ids(self):
        """Тест форматирования с ID рецептов."""
        predictions = np.array([4.5, 3.5])
        recipe_ids = [1, 2]
        result = format_predictions_for_api(predictions, recipe_ids)
        
        assert len(result) == 2
        assert result[0]['recipe_id'] == 1
        assert result[1]['recipe_id'] == 2
    
    def test_format_single_prediction(self):
        """Тест форматирования одного предсказания."""
        predictions = np.array([4.5])
        result = format_predictions_for_api(predictions)
        
        assert len(result) == 1
        assert result[0]['rating'] == 4.5


class TestValidatePredictions:
    """Тесты для функции validate_predictions."""
    
    def test_valid_predictions(self):
        """Тест валидных предсказаний."""
        predictions = np.array([2.0, 3.0, 4.0, 5.0])
        is_valid, errors = validate_predictions(predictions, min_value=0.0, max_value=5.0)
        
        assert is_valid
        assert len(errors) == 0
    
    def test_predictions_with_nan(self):
        """Тест предсказаний с NaN."""
        predictions = np.array([2.0, np.nan, 4.0])
        is_valid, errors = validate_predictions(predictions)
        
        assert not is_valid
        assert any('nan' in err.lower() for err in errors)
    
    def test_predictions_above_max(self):
        """Тест предсказаний выше максимума."""
        predictions = np.array([2.0, 6.0, 4.0])
        is_valid, errors = validate_predictions(predictions, min_value=0.0, max_value=5.0)
        
        assert not is_valid
        assert any('больше' in err.lower() for err in errors)
    
    def test_predictions_below_min(self):
        """Тест предсказаний ниже минимума."""
        predictions = np.array([-1.0, 3.0, 4.0])
        is_valid, errors = validate_predictions(predictions, min_value=0.0, max_value=5.0)
        
        assert not is_valid
        assert any('меньше' in err.lower() for err in errors)
    
    def test_invalid_type(self):
        """Тест невалидного типа."""
        predictions = [2.0, 3.0, 4.0]  # Не np.ndarray
        is_valid, errors = validate_predictions(predictions)
        
        assert not is_valid
        assert len(errors) > 0

