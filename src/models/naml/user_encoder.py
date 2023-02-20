# Adapted from https://github.com/yusanshi/news-recommendation/blob/master/src/model/NAML/user_encoder.py

import torch
import torch.nn as nn

from src.models.components.attention import AdditiveAttention


class UserEncoder(nn.Module):
    def __init__(
            self,
            num_filters: int,
            query_vector_dim: int
            ) -> None:

        super().__init__()

        self.additive_attention = AdditiveAttention(num_filters, query_vector_dim)

    def forward(self, clicked_news_vector: torch.Tensor) -> torch.Tensor:
        # batch_size, num_clicked_news_user, num_filters -> batch_size, num_filters
        user_vector = self.additive_attention(clicked_news_vector)

        return user_vector

