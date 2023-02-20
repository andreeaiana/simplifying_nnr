# Adapted from https://github.com/taoqi98/CAUM/blob/main/Code/Models.py

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.components.attention import AdditiveAttention


class TextEncoder(nn.Module):
    def __init__(
            self,
            pretrained_embeddings: torch.Tensor,
            embedding_dim: int,
            num_attention_heads: int,
            query_vector_dim: int,
            dropout_probability: float,
            ) -> None:
        
        super().__init__()
        
        self.embedding = nn.Embedding.from_pretrained(
                torch.FloatTensor(pretrained_embeddings),
                freeze=False,
                padding_idx=0
                )
        self.multihead_attention = nn.MultiheadAttention(
                embed_dim=embedding_dim, 
                num_heads=num_attention_heads
                )
        self.additive_attention = AdditiveAttention(
                input_dim=embedding_dim,
                query_dim=query_vector_dim
                )
        self.dropout = nn.Dropout(dropout_probability)

    def forward(self, text: torch.Tensor) -> torch.Tensor:
        # batch_size, num_words_text, word_embedding_dim
        text_vector = self.embedding(text)
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


class CategoryEncoder(nn.Module):
    def __init__(
            self,
            num_categories: int,
            category_embedding_dim: int,
            category_output_dim: int,
            dropout_probability: float
            ) -> None:

        super().__init__()

        self.category_embedding = nn.Embedding(
                num_embeddings=num_categories, 
                embedding_dim=category_embedding_dim, 
                padding_idx=0
                )
        self.linear = nn.Linear(category_embedding_dim, category_output_dim)
        self.dropout = nn.Dropout(dropout_probability)

    def forward(self, category: torch.Tensor) -> torch.Tensor:
        category_vector = self.category_embedding(category)
        category_vector = self.dropout(category_vector)
        category_vector = F.relu(self.linear(category_vector))

        return category_vector


class NewsEncoder(nn.Module):
    def __init__(
            self,
            pretrained_word_embeddings: torch.Tensor,
            pretrained_entity_embeddings: torch.Tensor,
            word_embedding_dim: int,
            entity_embedding_dim: int,
            num_attention_heads: int,
            query_vector_dim: int,
            num_categories: int,
            category_embedding_dim: int,
            news_vector_dim: int,
            dropout_probability: float,
            ) -> None:

        super().__init__()

        # text encoder
        self.text_encoder = TextEncoder(
                pretrained_embeddings=pretrained_word_embeddings,
                embedding_dim=word_embedding_dim,
                num_attention_heads=num_attention_heads,
                query_vector_dim=query_vector_dim,
                dropout_probability=dropout_probability
                )
        
        # entity encoder
        self.entity_encoder = TextEncoder(
                pretrained_embeddings=pretrained_entity_embeddings,
                embedding_dim=entity_embedding_dim,
                num_attention_heads=num_attention_heads,
                query_vector_dim=query_vector_dim,
                dropout_probability=dropout_probability
                )
        
        # category encoder
        self.category_encoder = CategoryEncoder(
                num_categories=num_categories,
                category_embedding_dim=category_embedding_dim,
                category_output_dim=category_embedding_dim,
                dropout_probability=dropout_probability
                )

        self.linear = nn.Linear(
                in_features=word_embedding_dim + entity_embedding_dim + category_embedding_dim,
                out_features=news_vector_dim
                )

    def forward(self, news: Dict[str, torch.Tensor]) -> torch.Tensor:
        text_vector = self.text_encoder(news['title'])
        entity_vector = self.entity_encoder(news['title_entities'])
        category_vector = self.category_encoder(news['category'])

        all_vectors = torch.cat(
                [text_vector, entity_vector, category_vector],
                dim=-1
                )

        news_vector = self.linear(all_vectors)

        return news_vector
