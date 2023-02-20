# Adapted from https://github.com/taoqi98/FedNewsRec/blob/master/code/models.py

import torch
import torch.nn as nn

from src.models.components.attention import AdditiveAttention


class UserEncoder(nn.Module):
    def __init__(
            self,
            num_filters: int,
            num_attention_heads: int,
            query_vector_dim: int,
            gru_hidden_dim: int,
            num_recent_news: int,
            dropout_probability: float
            ) -> None:
        super().__init__()

        self.num_recent_news = num_recent_news

        self.multihead_attention = nn.MultiheadAttention(num_filters, num_attention_heads)
        self.additive_attention = AdditiveAttention(num_filters, query_vector_dim)
        self.gru = nn.GRU(num_filters, gru_hidden_dim, batch_first=True)
        self.final_additive_attention = AdditiveAttention(gru_hidden_dim, query_vector_dim)
        self.dropout = nn.Dropout(p=dropout_probability)

    def forward(self, clicked_news_vector: torch.Tensor) -> torch.Tensor:
        # long-term user representation
        # batch_size, num_clicked_news_user, num_filters
        longterm_user_vector, _ = self.multihead_attention(
                clicked_news_vector, clicked_news_vector, clicked_news_vector
                )
        longterm_user_vector = self.dropout(longterm_user_vector)

        # batch_size, num_filters
        longterm_user_vector = self.additive_attention(longterm_user_vector)

        # short-term user repesentation
        # batch_size, num_recent_news, num_filters
        recent_clicked_news = clicked_news_vector[:, -self.num_recent_news:, :]

        # 1, batch_size, gru_hidden_dim
        _, hidden = self.gru(recent_clicked_news)

        # batch_size, gru_hidden_dim
        shortterm_user_vector = hidden.squeeze(dim=0)

        # aggregated user representation
        # batch_size, 2, gru_hidden_dim
        user_vector = torch.stack(
                [shortterm_user_vector, longterm_user_vector],
                dim=1)

        # batch_size, gru_hidden_dim
        user_vector = self.final_additive_attention(user_vector)

        return user_vector

