# Adapted from https://github.com/info-ruc/ai20projects/blob/ca6f993cfa569250b3116921f4b481d01de36197/2018202180/src/scripts/models/NPA.py

import torch 
import torch.nn as nn
import torch.nn.functional as F

from src.models.components.attention import PersonalizedAttention
from src.models.components.projections import UserPreferenceQueryProjection


class NewsEncoder(nn.Module):
    def __init__(
            self,
            pretrained_word_embeddings: torch.Tensor,
            word_embedding_dim: int,
            user_embedding_dim: int,
            num_filters: int,
            window_size: int,
            preference_query_dim: int,
            dropout_probability: float
            ) -> None:
        super().__init__()

        self.word_embedding = nn.Embedding.from_pretrained(
                embeddings=pretrained_word_embeddings,
                freeze=False,
                padding_idx=0
                )
        self.cnn = nn.Conv1d(
                in_channels=word_embedding_dim,
                out_channels=num_filters,
                kernel_size=window_size,
                padding=1
                )
        self.word_query_projection = UserPreferenceQueryProjection(
                user_embedding_dim=user_embedding_dim,
                preference_query_dim=preference_query_dim,
                dropout_probability=dropout_probability
                )
        self.personalized_attention = PersonalizedAttention(
                preference_query_dim=preference_query_dim,
                num_filters=num_filters
                )
        self.dropout = nn.Dropout(p=dropout_probability)

    def forward(self, text: torch.Tensor, lengths: torch.Tensor, projected_users: torch.Tensor) -> torch.Tensor:
        # batch_size, num_words_text, word_embedding_dim
        text_vector = self.word_embedding(text)
        text_vector = self.dropout(text_vector)

        # batch_size, num_filters, num_words_text
        text_vector = self.cnn(text_vector.permute(0, 2, 1))
        text_vector = F.relu(text_vector)
        text_vector = self.dropout(text_vector)

        # batch_size, query_preference_dim
        word_preference_query = self.word_query_projection(projected_users)
        word_preference_query = torch.repeat_interleave(word_preference_query, lengths, dim=0)
        
        # batch_size, num_filters
        news_vector = self.personalized_attention(word_preference_query, text_vector)

        return news_vector
