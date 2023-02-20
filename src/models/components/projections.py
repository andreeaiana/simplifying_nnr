# Adapted from https://github.com/info-ruc/ai20projects/blob/ca6f993cfa569250b3116921f4b481d01de36197/2018202180/src/scripts/models/NPA.py

import torch 
import torch.nn as nn
import torch.nn.functional as F


class UserProjection(nn.Module):
    """ Embeds user ID to dense vector through a lookup table."""
    def __init__(
            self,
            num_users: int,
            user_embedding_dim: int,
            dropout_probability: float
            ) -> None:
        super().__init__()

        self.user_embedding = nn.Parameter(torch.rand(num_users, user_embedding_dim))
        self.dropout = nn.Dropout(p=dropout_probability)

    def forward(self, users: torch.Tensor) -> torch.Tensor:
        """
        Args: 
            users (torch.Tensor): batch_size

        Returns:
            torch.Tensor: batch_size * user_embedding_dim
        """
        projected_users = self.user_embedding[users]
        projected_users = self.dropout(projected_users)

        return projected_users


class UserPreferenceQueryProjection(nn.Module):
    """ Projects dense user representations to preference query vector. """
    def __init__(
            self,
            user_embedding_dim: int,
            preference_query_dim: int,
            dropout_probability: float
            ) -> None:
        super().__init__()

        self.preference_query_projection = nn.Linear(user_embedding_dim, preference_query_dim)
        self.dropout = nn.Dropout(p=dropout_probability)

    def forward(self, projected_users: torch.Tensor) -> torch.Tensor:
        """
        Args:
            projected_users (torch.Tensor): batch_size * user_embedding_dim

        Returs:
            torch.Tensor: batch_size * preference_dim
        """
        query = self.preference_query_projection(projected_users)
        query = F.relu(query)
        query = self.dropout(query)

        return query
