from transformers import PreTrainedModel, PretrainedConfig
import torch
import torch.nn as nn

class RecipeRankerConfig(PretrainedConfig):
    model_type = "RecipeRanker"

    def __init__(
        self,
        vocab_size=5000,
        embedding_dim=64,
        num_numerical_features=3,
        hidden_dims=[128, 64],
        dropout_rate=0.3,
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_numerical_features = num_numerical_features
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        super().__init__(**kwargs)


class RecipeRankerModel(PreTrainedModel):
    config_class = RecipeRankerConfig

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        
        # Слой для обработки текстовых признаков (TF-IDF)
        # Он будет соединен с числовыми признаками, поэтому создаем MLP для их совместной обработки
        
        # Общий размер входа для MLP
        input_dim = config.vocab_size + config.num_numerical_features

        layers = []
        for hidden_dim in config.hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(config.dropout_rate))
            input_dim = hidden_dim
        
        self.mlp = nn.Sequential(*layers)
        
        # Выходной слой для регрессии (предсказание одного числа - рейтинга)
        self.regressor = nn.Linear(config.hidden_dims[-1], 1)

    def forward(self, numerical_features, text_features, **kwargs):
        # Объединяем числовые и текстовые признаки
        combined_features = torch.cat((numerical_features, text_features), dim=1)
        
        # Прогоняем через MLP
        hidden_state = self.mlp(combined_features)
        
        # Получаем итоговый прогноз
        logits = self.regressor(hidden_state)
        
        return logits
