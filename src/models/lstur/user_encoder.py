# Adapted from https://github.com/yusanshi/news-recommendation/blob/master/src/model/LSTUR/user_encoder.py

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class UserEncoder(nn.Module):
    def __init__(
            self,
            num_users: int,
            input_dim: int,
            input_dim_multiplier: int,
            user_masking_probability: float,
            long_short_term_method: str
            ) -> None:

        super().__init__()

        assert long_short_term_method in ['ini', 'con']
        self.long_short_term_method = long_short_term_method

        self.long_term_user_embedding = nn.Embedding(
                num_embeddings=num_users,
                embedding_dim=input_dim * input_dim_multiplier if self.long_short_term_method == 'ini' \
                        else int(input_dim * input_dim_multiplier * 0.5),
                padding_idx=0
                )
        self.dropout = nn.Dropout2d(p=user_masking_probability)
        self.gru = nn.GRU(
                input_dim * input_dim_multiplier,
                input_dim * input_dim_multiplier if self.long_short_term_method == 'ini' \
                        else int(input_dim * input_dim_multiplier * 0.5)
                )

    def forward(self, user: torch.Tensor, clicked_news_vector: torch.Tensor, hist_size: torch.Tensor) -> torch.Tensor:
        # long-term user representation
        user_vector = self.long_term_user_embedding(user).unsqueeze(dim=0)
        user_vector = self.dropout(user_vector)

        # short-term user representation
        packed_clicked_news_vector = pack_padded_sequence(
                input=clicked_news_vector,
                lengths=hist_size.cpu().int(),
                batch_first=True,
                enforce_sorted=False
                )
        if self.long_short_term_method == 'ini':
            _, last_hidden = self.gru(packed_clicked_news_vector, user_vector)
            return last_hidden.squeeze(dim=0)
        else:
            _, last_hidden = self.gru(packed_clicked_news_vector)
            return torch.cat(
                    (last_hidden.squeeze(dim=0), user_vector.squeeze(dim=0)), 
                    dim=1
                    )

