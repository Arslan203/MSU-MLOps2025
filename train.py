import argparse
import yaml
import logging
import torch
import numpy as np
import os

from src.utils import setup_logging
from src.data_loader import load_and_prepare_data
from src.model import RecipeRankerConfig, RecipeRankerModel
from src.trainer import train_model

def main(config_path):
    """Главная функция для запуска всего пайплайна."""
    setup_logging()
    
    logging.info(f"Загрузка конфигурации из {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # 1. Воспроизводимость
    seed = config['training']['random_seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 2. Загрузка и подготовка данных
    train_loader, val_loader = load_and_prepare_data(config)

    # 3. Инициализация модели
    # Конфиг модели создается на основе общего конфига
    model_config = RecipeRankerConfig(
        vocab_size=config['model']['vocab_size'],
        embedding_dim=config['model']['embedding_dim'],
        num_numerical_features=config['model']['num_numerical_features'],
        hidden_dims=config['model']['hidden_dims'],
        dropout_rate=config['model']['dropout_rate']
    )
    model = RecipeRankerModel(config=model_config)
    logging.info("Архитектура модели:")
    logging.info(model)
    
    # 4. Обучение модели
    logging.info("Начало обучения...")
    trained_model = train_model(model, train_loader, val_loader, config)

    # 5. Сохранение модели
    output_path = config['training']['output_model_path']
    os.makedirs(output_path, exist_ok=True)
    trained_model.save_pretrained(output_path)
    logging.info(f"Обученная модель сохранена в формате Hugging Face в {output_path}")

    # 6. Простая проверка
    # Загружаем модель обратно и делаем предсказание для одного батча
    logging.info("Проверка загрузки модели и предсказания...")
    loaded_model = RecipeRankerModel.from_pretrained(output_path)
    loaded_model.eval()
    
    sample_batch = next(iter(val_loader))
    with torch.no_grad():
        predictions = loaded_model(
            numerical_features=sample_batch['numerical_features'],
            text_features=sample_batch['text_features']
        )
    logging.info("Пример предсказаний для одного батча:")
    logging.info(predictions.squeeze().numpy()[:5])
    logging.info("Реальные значения:")
    logging.info(sample_batch['target'].squeeze().numpy()[:5])
    logging.info("Пайплайн успешно завершен!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Скрипт для обучения модели RecipeRanker.")
    parser.add_argument('--config', type=str, required=True, help="Путь к файлу конфигурации (YAML).")
    args = parser.parse_args()
    
    main(args.config)
