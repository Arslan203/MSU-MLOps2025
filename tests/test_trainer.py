"""Тесты для модуля trainer."""
import pytest
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from src.trainer import train_epoch, validate_epoch, get_training_metrics
from src.model import RecipeRankerConfig, RecipeRankerModel


@pytest.fixture
def dummy_model():
    """Создает фиктивную модель для тестов."""
    config = RecipeRankerConfig(
        vocab_size=10,
        embedding_dim=64,
        num_numerical_features=3,
        hidden_dims=[32, 16],
        dropout_rate=0.1
    )
    return RecipeRankerModel(config)


@pytest.fixture
def dummy_dataloader():
    """Создает фиктивный DataLoader для тестов."""
    batch_size = 4
    num_samples = 8
    
    numerical_features = torch.randn(num_samples, 3)
    text_features = torch.randn(num_samples, 10)
    targets = torch.rand(num_samples, 1) * 5  # Рейтинги от 0 до 5
    
    dataset = TensorDataset(numerical_features, text_features, targets)
    return DataLoader(dataset, batch_size=batch_size)


@pytest.fixture
def dummy_optimizer(dummy_model):
    """Создает фиктивный оптимизатор для тестов."""
    from torch.optim import AdamW
    return AdamW(dummy_model.parameters(), lr=0.001)


@pytest.fixture
def dummy_loss_fn():
    """Создает фиктивную функцию потерь для тестов."""
    import torch.nn as nn
    return nn.MSELoss()


class TestTrainEpoch:
    """Тесты для функции train_epoch."""
    
    def test_train_epoch_returns_loss(self, dummy_model, dummy_dataloader, 
                                      dummy_optimizer, dummy_loss_fn):
        """Тест что train_epoch возвращает loss."""
        device = torch.device('cpu')
        
        # Создаем правильный формат данных для DataLoader
        batch_size = 4
        num_samples = 8
        
        numerical_features = torch.randn(num_samples, 3)
        text_features = torch.randn(num_samples, 10)
        targets = torch.rand(num_samples, 1) * 5
        
        # Создаем кастомный Dataset
        class CustomDataset(torch.utils.data.Dataset):
            def __init__(self, num, text, target):
                self.num = num
                self.text = text
                self.target = target
            
            def __len__(self):
                return len(self.target)
            
            def __getitem__(self, idx):
                return {
                    'numerical_features': self.num[idx],
                    'text_features': self.text[idx],
                    'target': self.target[idx]
                }
        
        dataset = CustomDataset(numerical_features, text_features, targets)
        dataloader = DataLoader(dataset, batch_size=batch_size)
        
        loss = train_epoch(dummy_model, dataloader, dummy_optimizer, dummy_loss_fn, device)
        
        assert isinstance(loss, float)
        assert loss >= 0
    
    def test_train_epoch_updates_model(self, dummy_model, dummy_dataloader,
                                      dummy_optimizer, dummy_loss_fn):
        """Тест что train_epoch обновляет модель."""
        device = torch.device('cpu')
        
        # Сохраняем начальные веса
        initial_params = [p.clone() for p in dummy_model.parameters()]
        
        # Создаем правильный формат данных
        batch_size = 4
        num_samples = 8
        
        numerical_features = torch.randn(num_samples, 3)
        text_features = torch.randn(num_samples, 10)
        targets = torch.rand(num_samples, 1) * 5
        
        class CustomDataset(torch.utils.data.Dataset):
            def __init__(self, num, text, target):
                self.num = num
                self.text = text
                self.target = target
            
            def __len__(self):
                return len(self.target)
            
            def __getitem__(self, idx):
                return {
                    'numerical_features': self.num[idx],
                    'text_features': self.text[idx],
                    'target': self.target[idx]
                }
        
        dataset = CustomDataset(numerical_features, text_features, targets)
        dataloader = DataLoader(dataset, batch_size=batch_size)
        
        train_epoch(dummy_model, dataloader, dummy_optimizer, dummy_loss_fn, device)
        
        # Проверяем что веса изменились (по крайней мере некоторые)
        params_changed = any(
            not torch.equal(old, new)
            for old, new in zip(initial_params, dummy_model.parameters())
        )
        assert params_changed


class TestValidateEpoch:
    """Тесты для функции validate_epoch."""
    
    def test_validate_epoch_returns_metrics(self, dummy_model, dummy_dataloader, dummy_loss_fn):
        """Тест что validate_epoch возвращает метрики."""
        device = torch.device('cpu')
        
        # Создаем правильный формат данных
        batch_size = 4
        num_samples = 8
        
        numerical_features = torch.randn(num_samples, 3)
        text_features = torch.randn(num_samples, 10)
        targets = torch.rand(num_samples, 1) * 5
        
        class CustomDataset(torch.utils.data.Dataset):
            def __init__(self, num, text, target):
                self.num = num
                self.text = text
                self.target = target
            
            def __len__(self):
                return len(self.target)
            
            def __getitem__(self, idx):
                return {
                    'numerical_features': self.num[idx],
                    'text_features': self.text[idx],
                    'target': self.target[idx]
                }
        
        dataset = CustomDataset(numerical_features, text_features, targets)
        dataloader = DataLoader(dataset, batch_size=batch_size)
        
        # Убеждаемся что модель в режиме eval
        dummy_model.eval()
        
        val_loss, val_rmse, val_mae = validate_epoch(dummy_model, dataloader, dummy_loss_fn, device)
        
        # Метрики могут быть float или np.float32
        assert isinstance(val_loss, (float, np.floating))
        assert isinstance(val_rmse, (float, np.floating))
        assert isinstance(val_mae, (float, np.floating))
        assert val_loss >= 0
        assert val_rmse >= 0
        assert val_mae >= 0
        # Проверяем что метрики разумные (не NaN, не Inf)
        assert not np.isnan(float(val_loss))
        assert not np.isnan(float(val_rmse))
        assert not np.isnan(float(val_mae))
        assert not np.isinf(float(val_loss))
        assert not np.isinf(float(val_rmse))
        assert not np.isinf(float(val_mae))


class TestGetTrainingMetrics:
    """Тесты для функции get_training_metrics."""
    
    def test_get_training_metrics_returns_dict(self, dummy_model, dummy_dataloader):
        """Тест что get_training_metrics возвращает словарь метрик."""
        device = torch.device('cpu')
        
        # Создаем правильный формат данных
        batch_size = 4
        num_samples = 8
        
        numerical_features = torch.randn(num_samples, 3)
        text_features = torch.randn(num_samples, 10)
        targets = torch.rand(num_samples, 1) * 5
        
        class CustomDataset(torch.utils.data.Dataset):
            def __init__(self, num, text, target):
                self.num = num
                self.text = text
                self.target = target
            
            def __len__(self):
                return len(self.target)
            
            def __getitem__(self, idx):
                return {
                    'numerical_features': self.num[idx],
                    'text_features': self.text[idx],
                    'target': self.target[idx]
                }
        
        dataset = CustomDataset(numerical_features, text_features, targets)
        dataloader = DataLoader(dataset, batch_size=batch_size)
        
        metrics = get_training_metrics(dummy_model, dataloader, device)
        
        assert isinstance(metrics, dict)
        assert 'val_loss' in metrics
        assert 'val_rmse' in metrics
        assert 'val_mae' in metrics
        assert all(isinstance(v, float) for v in metrics.values())

