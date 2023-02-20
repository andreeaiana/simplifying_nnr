from typing import TypedDict, Optional, Any, Dict

import torch


class MINDBatch(TypedDict):
    batch_hist: torch.Tensor
    batch_cand: torch.Tensor
    x_hist: Dict[str, Any] 
    x_cand: Dict[str, Any]
    labels: Optional[torch.Tensor]
    users: torch.Tensor

