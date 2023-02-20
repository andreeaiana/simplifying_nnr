# https://github.com/whonor/MINS/blob/main/model/MINS/user_encoder.py

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

from src.models.components.attention import AdditiveAttention


class UserEncoder(nn.Module):
    def __init__(
            self,
            word_embedding_dim: int,
            query_vector_dim: int,
            num_filters: int,
            num_gru_channels: int
            ) -> None:
        super().__init__()
        
        self.num_gru_channels = num_gru_channels

        self.multihead_attention = nn.MultiheadAttention(
                embed_dim=word_embedding_dim,
                num_heads=num_gru_channels
                )
        self.additive_attention = AdditiveAttention(
                input_dim=word_embedding_dim,
                query_dim=query_vector_dim
                )

        assert num_filters % num_gru_channels == 0
        self.gru = nn.GRU(
                int(num_filters / num_gru_channels),
                int(num_filters / num_gru_channels)
                )
        self.multi_channel_gru = nn.ModuleList(
                [self.gru for _ in range(num_gru_channels)]
                )

    def forward(self, clicked_news_vector: torch.Tensor, hist_size: torch.Tensor) -> torch.Tensor:
        # batch_size, hist_size, word_embedding_dim
        multihead_user_vector, _ = self.multihead_attention(
                clicked_news_vector, clicked_news_vector, clicked_news_vector
                )

        # batch_size, hist_size, word_embedding_dim / num_gru_channels
        user_vector_channels = torch.chunk(
                input=multihead_user_vector,
                chunks=self.num_gru_channels,
                dim=2
                )
        channels = []
        for n, gru in zip(range(self.num_gru_channels), self.multi_channel_gru):
            packed_clicked_news_vector = pack_padded_sequence(
                input=user_vector_channels[n],
                lengths=hist_size.cpu().int(),
                batch_first=True,
                enforce_sorted=False
                )

            # 1, batch_size, num_filters / num_gru_channels
            _, last_hidden = gru(packed_clicked_news_vector)

            channels.append(last_hidden)

        # batch_size, 1, word_embedding_dim
        multi_channel_vector = torch.cat(channels, dim=2).transpose(0, 1)

        # batch_size, word_embedding_dim
        user_vector = self.additive_attention(multi_channel_vector)

        return user_vector

