# Adapted from https://github.com/yusanshi/news-recommendation/blob/master/src/model/DKN/KCNN.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class UserEncoder(nn.Module):
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            ) -> None:
       
        super().__init__()

        self.dnn = nn.Sequential(
                nn.Linear(
                    input_dim * 2,
                    hidden_dim
                    ),
                nn.Linear(hidden_dim, 1)
                )

    def forward(self, candidate_news_vector: torch.Tensor, clicked_news_vector: torch.Tensor, mask_cand: torch.Tensor, mask_hist: torch.Tensor) -> torch.Tensor:
        # num_clicked_news_user, batch_size, num_candidate_news_user, input_dim
        expanded_cand_news_vector = torch.repeat_interleave(
                input=candidate_news_vector.unsqueeze(dim=0),
                repeats=clicked_news_vector.shape[1],
                dim=0
                )

        # batch_size, num_candidate_news_user, num_clicked_news_user, input_dim
        expanded_cand_news_vector = expanded_cand_news_vector.permute(1, 2, 0, 3)
        
        # batch_size, num_candidate_news_user, num_clicked_news_user, input_dim
        repeated_clicked_news_vector = clicked_news_vector.unsqueeze(1).repeat(1, candidate_news_vector.shape[1], 1, 1)

        # batch_size, num_candidate_news_user, num_clicked_news_user, input_dim * 2
        concatenated_news_vectors = torch.cat(
                [expanded_cand_news_vector, repeated_clicked_news_vector],
                dim=-1
                )
        
        # batch_size, num_candidate_news_user, num_clicked_news_user
        transformed_news_vector = self.dnn(concatenated_news_vectors).squeeze(dim=-1)

        # num_clicked_news_user, batch_size, num_candidate_news_user
        repeated_mask_cand = torch.repeat_interleave(
                input=mask_cand.unsqueeze(0),
                repeats=mask_hist.shape[1],
                dim=0
                )

        # batch_size, num_candidate_news_user, num_clicked_news_user
        repeated_mask_cand = repeated_mask_cand.permute(1, 2, 0)

        # batch_size, num_candidate_news_user, num_clicked_news_user
        repeated_mask_hist = mask_hist.unsqueeze(dim=1).repeat(1, mask_cand.shape[1], 1)
        
        # softmax only for relevant clicked news for each user
        masked_transformed_news_vector = torch.where(
                ~repeated_mask_hist, 
                torch.tensor(
                    torch.finfo(transformed_news_vector.dtype).min,
                    device=transformed_news_vector.device
                    ),
                transformed_news_vector
                )

        # batch_size, num_candidate_news_user, num_clicked_news_user
        clicked_news_weights = F.softmax(masked_transformed_news_vector, dim=-1)
        
        # weights only for relevant candidate news for each user
        masked_clicked_news_weights = torch.where(
                ~repeated_mask_cand,
                torch.tensor(0., device=clicked_news_weights.device),
                clicked_news_weights
                )

        # batch_size, num_candidate_news_user, input_dim
        user_vector = torch.bmm(masked_clicked_news_weights, clicked_news_vector)

        return user_vector

