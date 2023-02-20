# Adapted from https://github.com/taoqi98/CAUM/blob/main/Code/Models.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.components.attention import DenseAttention


class UserEncoder(nn.Module):
    def __init__(
            self,
            news_vector_dim: int,
            num_filters: int,
            dense_att_hidden_dim1: int,
            dense_att_hidden_dim2: int,
            user_vector_dim: int,
            num_attention_heads: int,
            dropout_probability: float
            ) -> None:
        super().__init__()

        self.dropout1 = nn.Dropout(p=dropout_probability)
        self.dropout2 = nn.Dropout(p=dropout_probability)
        self.dropout3 = nn.Dropout(p=dropout_probability)

        self.linear1 = nn.Linear(news_vector_dim * 4, num_filters)
        self.linear2 = nn.Linear(news_vector_dim * 2, user_vector_dim)
        self.linear3 = nn.Linear(num_filters + user_vector_dim, user_vector_dim)

        self.dense_att = DenseAttention(
                input_dim=user_vector_dim * 2,
                hidden_dim1=dense_att_hidden_dim1,
                hidden_dim2=dense_att_hidden_dim2
                )
        self.multihead_attention = nn.MultiheadAttention(user_vector_dim, num_attention_heads)

    def forward(self, clicked_news_vector: torch.Tensor, cand_news_vector: torch.Tensor) -> torch.Tensor:
        cand_news_vector = self.dropout1(cand_news_vector)
        clicked_news_vector = self.dropout2(clicked_news_vector)

        repeated_cand_news_vector = cand_news_vector.unsqueeze(dim=1).repeat(1, clicked_news_vector.shape[1], 1)


        # candi-cnn
        clicked_news_left = torch.cat(
                [clicked_news_vector[:, -1:, :], clicked_news_vector[:, :-1, :]],
                dim=-2
                )
        clicked_news_right =  torch.cat(
                [clicked_news_vector[:, 1:, :], clicked_news_vector[:, :1, :]],
                dim=-2
                )
        clicked_news_cnn = torch.cat(
                [
                    clicked_news_left, 
                    clicked_news_vector, 
                    clicked_news_right, 
                    repeated_cand_news_vector
                    ],
                dim=-1
                )

        clicked_news_cnn = self.linear1(clicked_news_cnn)

        # candi-selfatt
        clicked_news = torch.cat(
                [repeated_cand_news_vector, clicked_news_vector],
                dim=-1) 
        clicked_news = self.linear2(clicked_news)
        clicked_news_self, _ = self.multihead_attention(
                clicked_news, 
                clicked_news, 
                clicked_news
                )

        clicked_news_all = torch.cat(
                [clicked_news_cnn, clicked_news_self], 
                dim=-1
                )
        clicked_news_all = self.dropout3(clicked_news_all)
        clicked_news_all = self.linear3(clicked_news_all)

        # candi-att
        attention_vector = torch.cat(
                [clicked_news_all, repeated_cand_news_vector],
                dim=-1)
        attention_score = self.dense_att(attention_vector)
        attention_score = attention_score.squeeze(dim=-1)
        attention_score = F.softmax(attention_score, dim=-1)

        user_vector = torch.bmm(attention_score.unsqueeze(dim=1), clicked_news_all).squeeze(dim=1)

        scores = torch.bmm(cand_news_vector.unsqueeze(dim=1), user_vector.unsqueeze(dim=-1)).flatten()

        return scores
