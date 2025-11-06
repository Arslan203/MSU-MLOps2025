import torch
import torch.nn as nn
from torch.optim import AdamW
import logging
import numpy as np
from typing import Dict


def train_epoch(model, train_loader, optimizer, loss_fn, device):
    """Выполняет одну эпоху обучения."""
    model.train()
    train_loss_total = 0.0

    for batch in train_loader:
        optimizer.zero_grad()

        numerical_features = batch["numerical_features"].to(device)
        text_features = batch["text_features"].to(device)
        targets = batch["target"].to(device)

        predictions = model(
            numerical_features=numerical_features, text_features=text_features
        )
        loss = loss_fn(predictions, targets)

        loss.backward()
        optimizer.step()

        train_loss_total += loss.item()

    avg_train_loss = train_loss_total / len(train_loader)
    return avg_train_loss


def validate_epoch(model, val_loader, loss_fn, device):
    """Выполняет валидацию на одной эпохе."""
    model.eval()
    val_loss_total = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in val_loader:
            numerical_features = batch["numerical_features"].to(device)
            text_features = batch["text_features"].to(device)
            targets = batch["target"].to(device)

            predictions = model(
                numerical_features=numerical_features, text_features=text_features
            )
            loss = loss_fn(predictions, targets)
            val_loss_total += loss.item()

            all_preds.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    avg_val_loss = val_loss_total / len(val_loader)
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    val_rmse = np.sqrt(avg_val_loss)  # Корень из MSE = RMSE
    val_mae = np.mean(np.abs(all_preds - all_targets))

    return avg_val_loss, val_rmse, val_mae


def train_model(model, train_loader, val_loader, config):
    """Функция для обучения и валидации модели."""
    device = torch.device(config["training"]["device"])
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=config["training"]["learning_rate"])
    loss_fn = nn.MSELoss()  # RMSE - это корень из MSE

    for epoch in range(config["training"]["epochs"]):
        # Обучение
        avg_train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)

        # Валидация
        avg_val_loss, val_rmse, val_mae = validate_epoch(
            model, val_loader, loss_fn, device
        )

        logging.info(
            f"Эпоха: {epoch + 1}/{config['training']['epochs']} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Val RMSE: {val_rmse:.4f} | "
            f"Val MAE: {val_mae:.4f}"
        )

    return model


def get_training_metrics(model, val_loader, device) -> Dict[str, float]:
    """Вычисляет метрики на валидационном наборе."""
    loss_fn = nn.MSELoss()
    avg_val_loss, val_rmse, val_mae = validate_epoch(model, val_loader, loss_fn, device)

    return {
        "val_loss": float(avg_val_loss),
        "val_rmse": float(val_rmse),
        "val_mae": float(val_mae),
    }
