import torch
import torch.nn as nn 


class DotProduct(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, user_vector: torch.Tensor, cand_news_vector: torch.Tensor) -> torch.Tensor:
        return torch.bmm(user_vector, cand_news_vector).squeeze(1)


class DNNPredictor(nn.Module):
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int
            ) -> None:
        super().__init__()
        
        self.dnn = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
                )

    def forward(self, user_vector: torch.Tensor, cand_news_vector: torch.Tensor) -> torch.Tensor:
        concat_vectors = torch.cat(
                [cand_news_vector.permute(0, 2, 1), user_vector], 
                dim=-1)

        return self.dnn(concat_vectors).squeeze(dim=-1)
