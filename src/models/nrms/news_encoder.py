# Adapted from https://github.com/yusanshi/news-recommendation/blob/master/src/model/NRMS/news_encoder.py

import torch
import torch.nn as nn

from src.models.components.attention import AdditiveAttention


class NewsEncoder(nn.Module):
    def __init__(
            self,
            pretrained_word_embeddings: torch.Tensor,
            word_embedding_dim: int,
            num_attention_heads: int,
            query_vector_dim: int,
            dropout_probability: float,
            ) -> None:
        
        super().__init__()
        
        self.word_embedding = nn.Embedding.from_pretrained(
                torch.FloatTensor(pretrained_word_embeddings),
                freeze=False,
                padding_idx=0
                )
        self.multihead_attention = nn.MultiheadAttention(
                embed_dim=word_embedding_dim, 
                num_heads=num_attention_heads
                )
        self.additive_attention = AdditiveAttention(
                input_dim=word_embedding_dim,
                query_dim=query_vector_dim
                )
        self.dropout = nn.Dropout(dropout_probability)

    def forward(self, text: torch.Tensor) -> torch.Tensor:
        # batch_size, num_words_text, word_embedding_dim
        text_vector = self.word_embedding(text)
        text_vector = self.dropout(text_vector)

        # num_words_text, batch_size, word_embedding_dim
        text_vector = text_vector.permute(1, 0, 2)
        text_vector, _ = self.multihead_attention(
                text_vector,
                text_vector,
                text_vector
                )
        text_vector = self.dropout(text_vector)
        
        # batch_size, word_embedding_dim
        text_vector = text_vector.permute(1, 0, 2)
        text_vector = self.additive_attention(text_vector)

        return text_vector
