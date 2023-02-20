# Adapted from https://github.com/taoqi98/FedNewsRec/blob/master/code/models.py

from typing import List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.components.attention import AdditiveAttention


class NewsEncoder(nn.Module):
    def __init__(
            self,
            pretrained_word_embeddings: torch.Tensor,
            word_embedding_dim: int,
            num_filters: int,
            window_size: int,
            num_attention_heads: int,
            query_vector_dim: int,
            dropout_probability: float
            ) -> None:
        super().__init__()
        
        self.word_embedding = nn.Embedding.from_pretrained(
                torch.FloatTensor(pretrained_word_embeddings),
                freeze=False,
                padding_idx=0
                )
        self.cnn = nn.Conv1d(
                in_channels=word_embedding_dim,
                out_channels=num_filters,
                kernel_size=window_size,
                padding=1
                )
        self.multihead_attention = nn.MultiheadAttention(
                embed_dim=num_filters,
                num_heads=num_attention_heads
                )
        self.additive_attention = AdditiveAttention(
                input_dim=num_filters,
                query_dim=query_vector_dim
                )
        self.dropout = nn.Dropout(p=dropout_probability)

    def forward(self, text: torch.Tensor) -> torch.Tensor:
        # batch_size, num_words_text, word_embedding_dim
        text_vector = self.word_embedding(text)
        text_vector = self.dropout(text_vector)

        # batch_size, num_filters, num_words_text
        text_vector = self.cnn(text_vector.permute(0, 2, 1))
        text_vector = F.relu(text_vector)
        text_vector = self.dropout(text_vector)

        # num_words_text, batch_size, num_filters
        text_vector = text_vector.permute(2, 0, 1)
        text_vector, _ = self.multihead_attention(
                text_vector, 
                text_vector, 
                text_vector
                )
        text_vector = self.dropout(text_vector)

        # batch_size, num_filters
        text_vector = self.additive_attention(text_vector.transpose(1, 0))

        return text_vector

