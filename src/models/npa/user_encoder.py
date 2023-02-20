# Adapted from https://github.com/info-ruc/ai20projects/blob/ca6f993cfa569250b3116921f4b481d01de36197/2018202180/src/scripts/models/NPA.py

import torch 
import torch.nn as nn

from src.models.components.attention import PersonalizedAttention
from src.models.components.projections import UserPreferenceQueryProjection


class UserEncoder(nn.Module):
    def __init__(
            self,
            user_embedding_dim: int,
            num_filters: int,
            preference_query_dim: int,
            dropout_probability: float
            ) -> None:
        super().__init__()

        self.news_query_projection = UserPreferenceQueryProjection(
                user_embedding_dim=user_embedding_dim,
                preference_query_dim=preference_query_dim,
                dropout_probability=dropout_probability
                )
        self.personalized_attention = PersonalizedAttention(
                preference_query_dim=preference_query_dim,
                num_filters=num_filters
                )

    def forward(self, clicked_news_vector: torch.Tensor, projected_users: torch.Tensor) -> torch.Tensor:
        # batch_size, query_preference_dim
        news_preference_query = self.news_query_projection(projected_users) 

        # batch_size, num_filters
        user_vector = self.personalized_attention(news_preference_query, clicked_news_vector.permute(0, 2, 1))

        return user_vector
