from typing import Any, List, Tuple

import torch
import numpy as np
from torch.nn import CrossEntropyLoss
from torch_geometric.utils import to_dense_batch
from pytorch_lightning import LightningModule
from torchmetrics.classification import AUROC
from torchmetrics.retrieval import RetrievalMRR, RetrievalNormalizedDCG
from torchmetrics import MeanMetric, MinMetric, MetricCollection

from src.datamodules.components.mind_batch import MINDBatch
from src.models.dkn.news_encoder import NewsEncoder
from src.models.dkn.user_encoder import UserEncoder
from src.models.components.click_predictors import DotProduct, DNNPredictor
from src.models.components.losses import SupConLoss


class DKNModule(LightningModule):
    def __init__(
            self,
            supcon_loss: bool,
            late_fusion: bool,
            temperature: float,
            use_context: bool,
            pretrained_word_embeddings_path: str,
            pretrained_entity_embeddings_path: str,
            word_embedding_dim: int,
            entity_embedding_dim: int,
            num_filters: int,
            window_sizes: List[int],
            hidden_dim_dnn: int,
            optimizer: torch.optim.Optimizer
            ) -> None:
        
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # load pretrained word embeddings
        pretrained_word_embeddings = torch.from_numpy(np.load(self.hparams.pretrained_word_embeddings_path)).float()

        # load pretrained entity embeddings
        pretrained_entity_embeddings = torch.from_numpy(np.load(self.hparams.pretrained_entity_embeddings_path)).float()
        pretrained_context_embeddings = torch.from_numpy(np.load(self.hparams.pretrained_entity_embeddings_path)).float()

        # model components
        self.news_encoder = NewsEncoder(
                use_context=self.hparams.use_context,
                pretrained_word_embeddings=pretrained_word_embeddings,
                pretrained_entity_embeddings=pretrained_entity_embeddings,
                pretrained_context_embeddings=pretrained_context_embeddings,
                word_embedding_dim=self.hparams.word_embedding_dim,
                entity_embedding_dim=self.hparams.entity_embedding_dim,
                num_filters=self.hparams.num_filters,
                window_sizes=self.hparams.window_sizes
                )

        # user encoder 
        if not self.hparams.late_fusion:
            # initialize user encoder if early fusion
            self.user_encoder = UserEncoder(
                    input_dim=len(self.hparams.window_sizes) * self.hparams.num_filters,
                    hidden_dim=self.hparams.hidden_dim_dnn
                    )

        # click predictor
        if not self.hparams.late_fusion:
            self.click_predictor = DNNPredictor(
                    input_dim=len(self.hparams.window_sizes) * 2 * self.hparams.num_filters,
                    hidden_dim=self.hparams.hidden_dim_dnn
                    )
        else:
            self.click_predictor = DotProduct()
      
        # loss fuction
        if not self.hparams.supcon_loss:
            self.criterion = CrossEntropyLoss()
        else:
            self.criterion = SupConLoss(temperature=self.hparams.temperature)
        
        # metric objects for calculating and averaging performance across batches        
        metrics = MetricCollection({
            'auc': AUROC(task='binary', num_classes=2),
            'mrr': RetrievalMRR(),
            'ndcg@5': RetrievalNormalizedDCG(k=5),
            'ndcg@10': RetrievalNormalizedDCG(k=10)
            })
        self.train_metrics = metrics.clone(prefix='train/')
        self.val_metrics = metrics.clone(prefix='val/')
        self.test_metrics = metrics.clone(prefix='test/')

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation loss
        self.val_loss_best = MinMetric()
        
    def forward(self, batch: MINDBatch) -> torch.Tensor:
        # encode user history
        clicked_news_vector = self.news_encoder(batch['x_hist'])
        clicked_news_vector_agg, mask_hist = to_dense_batch(clicked_news_vector, batch['batch_hist'])

        # encode candidate news
        candidate_news_vector = self.news_encoder(batch['x_cand'])
        candidate_news_vector_agg, mask_cand = to_dense_batch(candidate_news_vector, batch['batch_cand'])
        
        # encode user
        if not self.hparams.late_fusion:
            # learn user encoder
            user_vector = self.user_encoder(candidate_news_vector_agg, clicked_news_vector_agg, mask_cand, mask_hist)
        else:
            # aggregate embeddings of clicked news
            hist_size = torch.tensor([torch.where(mask_hist[i])[0].shape[0] for i in range(mask_hist.shape[0])], device=self.device)
            user_vector = torch.div(
                    clicked_news_vector_agg.sum(dim=1),
                    hist_size.unsqueeze(dim=-1)
                    ) 
            user_vector = user_vector.unsqueeze(dim=1)
        
        # click scores for candidate news
        scores = self.click_predictor(user_vector, candidate_news_vector_agg.permute(0, 2, 1))
        if not self.hparams.late_fusion:
            scores = torch.where(~mask_cand, torch.tensor(0., device=self.device), scores)

        return scores

    def on_train_start(self) -> None:
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_acc_best doesn't store accuracy from these checks
        self.val_loss_best.reset()

    def model_step(self, batch: MINDBatch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        scores = self(batch)
        y_true, mask_cand = to_dense_batch(batch['labels'], batch['batch_cand'])

        if not self.hparams.supcon_loss:
            loss = self.criterion(scores, y_true)
        else:
            # indices of positive pairs for loss calculation
            pos_idx = [torch.where(y_true[i])[0] for i in range(mask_cand.shape[0])]
            pos_repeats = torch.tensor([len(pos_idx[i]) for i in range(len(pos_idx))])
            q_p = torch.repeat_interleave(torch.arange(mask_cand.shape[0]), pos_repeats)
            p = torch.cat(pos_idx)

            # indices of negative pairs for loss calculation
            neg_idx = [torch.where(~y_true[i].bool())[0][:len(torch.where(mask_cand[i])[0])-pos_repeats[i]] for i in range(mask_cand.shape[0])]
            neg_repeats = torch.tensor([len(t) for t in neg_idx])
            q_n = torch.repeat_interleave(torch.arange(mask_cand.shape[0]), neg_repeats)
            n = torch.cat(neg_idx)

            indices_tuple = (q_p, p, q_n, n)
            loss = self.criterion(
                    embeddings=scores,
                    labels=None,
                    indices_tuple=indices_tuple,
                    ref_emb=None,
                    ref_labels=None
                    )
        
        # predictions, targets, indexes for metric computation
        preds = torch.cat([scores[n][mask_cand[n]] for n in range(mask_cand.shape[0])], dim=0).detach()
        targets = torch.cat([y_true[n][mask_cand[n]] for n in range(mask_cand.shape[0])], dim=0).long() 
        cand_news_size = torch.tensor([torch.where(mask_cand[n])[0].shape[0] for n in range(mask_cand.shape[0])])

        return loss, preds, targets, cand_news_size

    def training_step(self, batch: MINDBatch, batch_idx: int):
        loss, preds, targets, cand_news_size = self.model_step(batch)

        # update and log loss
        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or backpropagation will fail!
        return {
                "loss": loss, 
                "preds": preds, 
                "targets": targets, 
                "cand_news_size": cand_news_size
                }

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        preds = torch.cat([o['preds'] for o in outputs])
        targets = torch.cat([o['targets'] for o in outputs])
        cand_news_size = torch.cat([o['cand_news_size'] for o in outputs])
        indexes = torch.arange(cand_news_size.shape[0]).repeat_interleave(cand_news_size)
        
        self.train_metrics(preds, targets, **{'indexes': indexes})
        self.log_dict(self.train_metrics, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def validation_step(self, batch: MINDBatch, batch_idx: int):
        loss, preds, targets, cand_news_size = self.model_step(batch)

        # update and log loss
        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return {
                "loss": loss, 
                "preds": preds, 
                "targets": targets, 
                "cand_news_size": cand_news_size
                }

    def validation_epoch_end(self, outputs: List[Any]):
        loss = self.val_loss.compute()  # get current val loss
        self.val_loss_best(loss)  # update best so far val loss

        # log `val_loss_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/loss_best", self.val_loss_best.compute(), prog_bar=True, logger=True, sync_dist=True)

        preds = torch.cat([o['preds'] for o in outputs])
        targets = torch.cat([o['targets'] for o in outputs])
        cand_news_size = torch.cat([o['cand_news_size'] for o in outputs])
        indexes = torch.arange(cand_news_size.shape[0]).repeat_interleave(cand_news_size)
        
        self.val_metrics(preds, targets, **{'indexes': indexes})
        self.log_dict(self.val_metrics, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch: MINDBatch, batch_idx: int):
        loss, preds, targets, cand_news_size = self.model_step(batch)

        # update and log loss
        self.test_loss(loss)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return {
                "loss": loss, 
                "preds": preds, 
                "targets": targets, 
                "cand_news_size": cand_news_size
                }

    def test_epoch_end(self, outputs: List[Any]):
        preds = torch.cat([o['preds'] for o in outputs])
        targets = torch.cat([o['targets'] for o in outputs])
        cand_news_size = torch.cat([o['cand_news_size'] for o in outputs])
        indexes = torch.arange(cand_news_size.shape[0]).repeat_interleave(cand_news_size)
        
        self.test_metrics(preds, targets, **{'indexes': indexes})
        self.log_dict(self.test_metrics, on_step=False, on_epoch=True, prog_bar=True, logger=True)
    
    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        """
        optimizer = self.hparams.optimizer(params=self.parameters())
        
        return {"optimizer": optimizer}


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "model" / "dkn.yaml")
    _ = hydra.utils.instantiate(cfg)

