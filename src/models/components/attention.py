import torch
import torch.nn as nn
import torch.nn.functional as F


class AdditiveAttention(nn.Module):
    def __init__(self, input_dim: int, query_dim: int) -> None:
        super().__init__()

        self.linear = nn.Linear(input_dim, query_dim)
        self.query = nn.Parameter(
                torch.empty(query_dim).uniform_(-0.1, 0.1))

    def forward(self, input_vector: torch.Tensor) -> torch.Tensor:
        """ 
        Args: 
            input_vector (torch.Tensor): User tensor of shape (batch_size, hidden_dim, output_dim)

        Returns:
            torch.Tensor: User tensor of shape (batch_size, news_emb_dim).
        """
        # batch_size, hidden_dim, output_dim
        attention = torch.tanh(self.linear(input_vector))

        # batch_size, hidden_dim
        attention_weights = F.softmax(torch.matmul(attention, self.query), dim=1)

        # batch_size, output_dim
        weighted_input = torch.bmm(
                attention_weights.unsqueeze(dim=1), input_vector
                ).squeeze(dim=1)
        
        return weighted_input


class PersonalizedAttention(nn.Module):
    """ Personalized attention from NPA."""
    def __init__(
            self,
            preference_query_dim: int,
            num_filters: int
            ) -> None:
        super().__init__()

        self.preference_query_projection = nn.Linear(preference_query_dim, num_filters)

    def forward(self, query: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query (torch.Tensor): batch_size * preference_dim
            keys (torch.Tensor): batch_size * num_filters * num_words_text

        Returns:
            torch.Tensor: batch_size * num_filters
        """
        # batch_size * 1 * num_filters
        query = torch.tanh(self.preference_query_projection(query).unsqueeze(dim=1))

        # batch_size * 1 * num_words_text
        attn_results = torch.bmm(query, keys)

        # batch_size * num_words_text * 1
        attn_weights = F.softmax(attn_results, dim=2).permute(0, 2, 1)

        # batch_size * num_filters
        attn_aggr = torch.bmm(keys, attn_weights).squeeze()

        return attn_aggr


class DenseAttention(nn.Module):
    def __init__(
            self,
            input_dim: int,
            hidden_dim1: int,
            hidden_dim2: int
            ) -> None:
        super().__init__()

        self.linear = nn.Linear(input_dim, hidden_dim1)
        self.tanh1 = nn.Tanh()
        self.linear2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.tanh2 = nn.Tanh()
        self.linear3 = nn.Linear(hidden_dim2, 1)

    def forward(self, input_vector: torch.Tensor) -> torch.Tensor:
        transformed_vector = self.linear(input_vector)
        transformed_vector = self.tanh1(transformed_vector)
        transformed_vector = self.linear2(transformed_vector)
        transformed_vector = self.tanh2(transformed_vector)
        transformed_vector = self.linear3(transformed_vector)

        return transformed_vector
