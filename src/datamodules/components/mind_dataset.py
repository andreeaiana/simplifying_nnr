from dataclasses import dataclass
from typing import Any, List, Dict, Tuple, Sequence

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd

from src.datamodules.components.mind_dataframe import MINDDataFrame
from src.datamodules.components.mind_batch import MINDBatch

class MINDDatasetTrain(MINDDataFrame):
    def __init__(self, news: pd.DataFrame, behaviors: pd.DataFrame, max_history_length: int, neg_sampling_ratio: int) -> None:
        self.news = news
        self.behaviors = behaviors
        self.max_history_length = max_history_length
        self.neg_sampling_ratio = neg_sampling_ratio

    def __len__(self) -> int:
        return len(self.behaviors)

    def __getitem__(self, index: Any) -> Tuple[np.ndarray, pd.DataFrame, pd.DataFrame, np.ndarray]:
        bhv = self.behaviors.iloc[index]

        user = np.array([int(bhv['user'])])
        history = np.array(bhv['history'])[:self.max_history_length]
        candidates = np.array(bhv['candidates'])
        labels = np.array(bhv['labels'])
        
        candidates, labels = self._sample_candidates(candidates, labels)
        
        history = self.news.loc[history]
        candidates = self.news.loc[candidates]

        return user, history, candidates, labels
            
    def _sample_candidates(self, candidates: np.ndarray, labels: np.ndarray) -> Tuple[List, List]:
        """Negative sampling of news candidates.
        
        Args: 
            candidates (np.array): Candidate news.
            labels (np.array): Labels of candidate news.

        Returns:
            - List: Sampled candidates.
            - np.array: Bool labels of sampled candidates (e.g. True if candidate was clicked, False otherwise)
        """
        pos_ids = np.where(labels==1)[0]
        neg_ids = np.array([]).astype(int)
       
        # sample with replacement when the candidates set is smaller than the negative sampling ratio
        replace_flag = True if (self.neg_sampling_ratio * len(pos_ids) > len(labels) - len(pos_ids)) else False
        
        # do negative sampling with the specified ratio
        neg_ids = np.random.choice(
            np.random.permutation(np.where(labels==0)[0]),
            self.neg_sampling_ratio * len(pos_ids),
            replace = replace_flag
            )
       
        indices = np.concatenate((pos_ids, neg_ids))
        indices = np.random.permutation(indices)
        candidates = candidates[indices]
        labels = labels[indices]

        return candidates, labels


class MINDDatasetTest(MINDDataFrame):
    def __init__(self, news: pd.DataFrame, behaviors: pd.DataFrame, max_history_length: int) -> None:
        self.news = news
        self.behaviors = behaviors
        self.max_history_length = max_history_length

    def __len__(self) -> int:
        return len(self.behaviors)

    def __getitem__(self, index: Any) -> Tuple[np.ndarray, pd.DataFrame, pd.DataFrame, np.ndarray]:
        bhv = self.behaviors.iloc[index]

        user = np.array([int(bhv['user'])])
        history = np.array(bhv['history'])[:self.max_history_length]
        candidates = np.array(bhv['candidates'])
        labels = np.array(bhv['labels'])

        history = self.news.loc[history]
        candidates = self.news.loc[candidates]

        return user, history, candidates, labels

@dataclass
class MINDCollate:
    def __init__(self, max_title_length: int, max_abstract_length: int) -> None:
        self.max_title_length = max_title_length
        self.max_abstract_length = max_abstract_length

    def __call__(self, batch):
        users, histories, candidates, labels = zip(*batch)
        
        batch_hist = self._make_batch_assignees(histories)
        batch_cand = self._make_batch_assignees(candidates)

        x_hist = self._tokenize_df(pd.concat(histories))
        x_cand = self._tokenize_df(pd.concat(candidates))
        labels = torch.from_numpy(np.concatenate(labels)).float()
        users = torch.from_numpy(np.concatenate(users)).long()
       
        return MINDBatch(
                batch_hist=batch_hist,
                batch_cand=batch_cand,
                x_hist=x_hist,
                x_cand=x_cand,
                labels=labels,
                users=users
                )

    def _tokenize(self, sequences: List[int], max_len: int) -> torch.Tensor:
        sequences_padded = [F.pad(torch.tensor(item), (0, max_len - len(item)), 'constant', 0) for item in sequences]
        return torch.vstack(sequences_padded).long()

    def _tokenize_df(self, df: pd.DataFrame) -> Dict[str, Any]:
        return {
                'title': self._tokenize(df['title'].values.tolist(), self.max_title_length),
                'abstract': self._tokenize(df['abstract'].values.tolist(), self.max_abstract_length), 
                'title_entities': self._tokenize(df['title_entities'].values.tolist(), self.max_title_length),
                'abstract_entities': self._tokenize(df['abstract_entities'].values.tolist(), self.max_abstract_length),
                'category': torch.from_numpy(df['category'].values).long(),
                'subcategory': torch.from_numpy(df['subcategory'].values).long()
                }
                 
    def _make_batch_assignees(self, items: Sequence[Sequence[Any]]) -> torch.Tensor:
        sizes = torch.tensor([len(x) for x in items])
        batch = torch.repeat_interleave(torch.arange(len(items)), sizes)
        return batch
