# Adapted from https://github.com/yusanshi/news-recommendation/blob/master/src/model/NRMS/user_encoder.py

import torch
import torch.nn as nn

from src.models.components.attention import AdditiveAttention


class UserEncoder(nn.Module):
    def __init__(
            self,
            word_embedding_dim: int,
            num_attention_heads: int,
            query_vector_dim: int
            ) -> None:

        super().__init__()

        self.multihead_attention = nn.MultiheadAttention(word_embedding_dim, num_attention_heads)
        self.additive_attention = AdditiveAttention(word_embedding_dim, query_vector_dim)

    def forward(self, clicked_news_vector: torch.Tensor) -> torch.Tensor:
        # batch_size, num_clicked_news_user, word_embeding_dim
        user_vector, _ = self.multihead_attention(
                clicked_news_vector, clicked_news_vector, clicked_news_vector
                )

        # batch_size, word_embeding_dim
        user_vector = self.additive_attention(user_vector)

        return user_vector
