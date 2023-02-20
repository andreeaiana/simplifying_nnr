# Adapted from https://github.com/yusanshi/news-recommendation/blob/master/src/model/NAML/news_encoder.py

from typing import List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.components.attention import AdditiveAttention


class TextEncoder(nn.Module):
    def __init__(
            self,
            pretrained_word_embeddings: torch.Tensor,
            word_embedding_dim: int,
            num_filters: int,
            window_size: int,
            query_vector_dim: int,
            dropout_probability: float,
            ) -> None:
        
        super().__init__()
        
        self.word_embedding = nn.Embedding.from_pretrained(
                torch.FloatTensor(pretrained_word_embeddings),
                freeze=False,
                padding_idx=0
                )
        self.cnn = nn.Conv2d(
                in_channels=1,
                out_channels=num_filters,
                kernel_size=(window_size, word_embedding_dim),
                padding=(int((window_size - 1) / 2), 0)
                )
        self.additive_attention = AdditiveAttention(
                input_dim=num_filters,
                query_dim=query_vector_dim
                )
        self.dropout = nn.Dropout(dropout_probability)

    def forward(self, text: torch.Tensor) -> torch.Tensor:
        # batch_size, num_words_text, word_embedding_dim
        text_vector = self.word_embedding(text)
        text_vector = self.dropout(text_vector)

        # batch_size, num_filters, num_words_text
        text_vector = self.cnn(text_vector.unsqueeze(dim=1)).squeeze(dim=3)
        text_vector = F.relu(text_vector)
        text_vector = self.dropout(text_vector)
        
        # batch_size, num_filters
        text_vector = self.additive_attention(text_vector.transpose(1,2))

        return text_vector


class CategoryEncoder(nn.Module):
    def __init__(
            self,
            num_categories: int,
            category_embedding_dim: int,
            category_output_dim: int
            ) -> None:

        super().__init__()

        self.category_embedding = nn.Embedding(
                num_embeddings=num_categories, 
                embedding_dim=category_embedding_dim, 
                padding_idx=0
                )
        self.linear = nn.Linear(category_embedding_dim, category_output_dim)

    def forward(self, category: torch.Tensor) -> torch.Tensor:
        return F.relu(self.linear(self.category_embedding(category)))


class NewsEncoder(nn.Module):
    def __init__(
            self,
            dataset_attributes: List[str],
            pretrained_word_embeddings: torch.Tensor,
            word_embedding_dim: int,
            num_filters: int,
            window_size: int,
            query_vector_dim: int,
            dropout_probability: float,
            num_categories: int,
            category_embedding_dim: int,
            ) -> None:

        super().__init__()

        assert len(dataset_attributes) > 0

        # text encoders
        text_encoder_candidates = ['title', 'abstract']
        self.text_encoders = nn.ModuleDict({
            name: 
            TextEncoder(
                pretrained_word_embeddings=pretrained_word_embeddings,
                word_embedding_dim=word_embedding_dim,
                num_filters=num_filters,
                window_size=window_size,
                query_vector_dim=query_vector_dim,
                dropout_probability=dropout_probability
                )
            for name in (set(dataset_attributes) & set(text_encoder_candidates))
            })

        # category encoders
        category_encoder_candidates = ['category', 'subcategory']
        self.category_encoders = nn.ModuleDict({
            name:
            CategoryEncoder(
                num_categories=num_categories,
                category_embedding_dim=category_embedding_dim,
                category_output_dim=num_filters
                )
            for name in (set(dataset_attributes) & set(category_encoder_candidates))
            })

        if len(dataset_attributes) > 1:
            self.additive_attention = AdditiveAttention(num_filters, query_vector_dim)

    def forward(self, news: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            news:
                {
                    "title": batch_size * num_words_title,
                    "abstract": batch_size * num_words_abstract,
                    "category": batch_size,
                    "subcategory": batch_size
                }
        Returns:
            (shape) batch_size, num_filters
        """
        text_vectors = [encoder(news[name]) for name, encoder in self.text_encoders.items()]
        category_vectors = [encoder(news[name]) for name, encoder in self.category_encoders.items()]

        all_vectors = text_vectors + category_vectors

        if len(all_vectors) == 1:
            final_news_vector = all_vectors[0]
        else:
            final_news_vector = self.additive_attention(
                    torch.stack(all_vectors, dim=1)
                    )

        return final_news_vector

