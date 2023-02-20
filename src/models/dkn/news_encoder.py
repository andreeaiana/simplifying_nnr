# Adapted from https://github.com/yusanshi/news-recommendation/blob/master/src/model/DKN/KCNN.py

from typing import List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class NewsEncoder(nn.Module):
    """ 
    Knowledge-aware CNN (KCNN) based on Kim CNN.
    """
    def __init__(
            self,
            use_context: bool,
            pretrained_word_embeddings: torch.Tensor,
            pretrained_entity_embeddings: torch.Tensor,
            pretrained_context_embeddings: torch.Tensor,
            word_embedding_dim: int,
            entity_embedding_dim: int,
            num_filters: int,
            window_sizes: List[int]
            ) -> None:

        super().__init__()

        self.use_context = use_context
        self.window_sizes = window_sizes

        self.word_embedding = nn.Embedding.from_pretrained(
                pretrained_word_embeddings, 
                freeze=False, 
                padding_idx=0
                )
        self.entity_embedding = nn.Embedding.from_pretrained(
                pretrained_entity_embeddings, 
                freeze=False, 
                padding_idx=0
                )
        if self.use_context:
            self.context_embedding = nn.Embedding.from_pretrained(
                    pretrained_context_embeddings, 
                    freeze=False, 
                    padding_idx=0
                    )

        self.transform_matrix = nn.Parameter(torch.empty(entity_embedding_dim, word_embedding_dim).uniform_(-0.1, 0.1))
        self.transform_bias = nn.Parameter(torch.empty(word_embedding_dim).uniform_(-0.1, 0.1))

        self.conv_filters = nn.ModuleDict({
            str(x): nn.Conv2d(
                3 if self.use_context else 2,
                num_filters,
                (x, word_embedding_dim)
                )
            for x in self.window_sizes
            })

    def forward(self, news: Dict[str, torch.Tensor]) -> torch.Tensor:
        # batch_size, num_words_text, word_embedding_dim
        word_vector = self.word_embedding(news['title'])

        # batch_size, num_words_text, entity_embedding_dim
        entity_vector = self.entity_embedding(news['title_entities'])

        # batch_size, num_words_text, word_embedding_dim
        transformed_entity_vector = torch.tanh(
                torch.add(
                    torch.matmul(entity_vector, self.transform_matrix),
                    self.transform_bias
                    )
                )

        if self.use_context:
            # batch_size, num_words_text, entity_embedding_dim
            context_vector = self.context_embedding(news['title_entities'])

            # batch_size, num_words_text, entity_embedding_dim
            transformed_context_vector = torch.tanh(
                    torch.add(
                        torch.matmul(context_vector, self.transform_matrix),
                        self.transform_bias
                        )
                    )
            
            # batch_size, 3, num_words_text, word_embedding_dim
            multi_channel_vector = torch.stack(
                    [word_vector, transformed_entity_vector, transformed_context_vector],
                    dim=1
                    )

        else:
            # batch_size, 2, num_words_text, word_embedding_dim
            multi_channel_vector = torch.stack(
                    [word_vector, transformed_entity_vector],
                    dim=1
                    )

        pooled_vectors = []
        for size in self.window_sizes:
            # batch_size, num_filters, num_words_text + 1 - size
            convoluted = self.conv_filters[str(size)](multi_channel_vector).squeeze(dim=3)
            activated = F.relu(convoluted)

            # batch_size, num_filters
            pooled = activated.max(dim=-1)[0]

            pooled_vectors.append(pooled)

        #batch_size, len(window_sizes) * num_filters
        news_vector = torch.cat(pooled_vectors, dim=1)

        return news_vector
