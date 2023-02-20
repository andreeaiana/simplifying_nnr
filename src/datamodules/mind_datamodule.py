from typing import Any, Dict, List, Optional

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from src.datamodules.components.mind_dataframe import MINDDataFrame
from src.datamodules.components.mind_dataset import MINDDatasetTrain, MINDDatasetTest, MINDCollate


class MINDDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        size: str,
        mind_urls: Dict[str, str],
        word_embeddings_url: str,
        word_embeddings_dirname: str,
        word_embeddings_fpath: str,
        entity_embeddings_filename: str,
        id2index_filenames: Dict[str, str],
        dataset_attributes: List[str],
        word_embedding_dim: int,
        entity_embedding_dim: int,
        entity_freq_threshold: int,
        entity_confidence_threshold: float,
        max_title_length: int,
        max_abstract_length: int,
        max_history_length: int,
        neg_sampling_ratio: int,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        drop_last: bool,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def prepare_data(self):
        """Download data if needed.

        Do not use it to assign state (self.x = y).
        """
        MINDDataFrame(
                data_dir=self.hparams.data_dir, 
                size=self.hparams.size,
                mind_urls=self.hparams.mind_urls,
                word_embeddings_url=self.hparams.word_embeddings_url,
                word_embeddings_dirname=self.hparams.word_embeddings_dirname,
                word_embeddings_fpath=self.hparams.word_embeddings_fpath,
                entity_embeddings_filename=self.hparams.entity_embeddings_filename,
                id2index_filenames=self.hparams.id2index_filenames,
                dataset_attributes=self.hparams.dataset_attributes,
                word_embedding_dim=self.hparams.word_embedding_dim,
                entity_embedding_dim=self.hparams.entity_embedding_dim,
                entity_freq_threshold=self.hparams.entity_freq_threshold,
                entity_confidence_threshold=self.hparams.entity_confidence_threshold,
                train=True,
                validation=False,
                download=True
                )
        MINDDataFrame(
                data_dir=self.hparams.data_dir, 
                size=self.hparams.size,
                mind_urls=self.hparams.mind_urls,
                word_embeddings_url=self.hparams.word_embeddings_url,
                word_embeddings_dirname=self.hparams.word_embeddings_dirname,
                word_embeddings_fpath=self.hparams.word_embeddings_fpath,
                entity_embeddings_filename=self.hparams.entity_embeddings_filename,
                id2index_filenames=self.hparams.id2index_filenames,
                dataset_attributes=self.hparams.dataset_attributes,
                word_embedding_dim=self.hparams.word_embedding_dim,
                entity_embedding_dim=self.hparams.entity_embedding_dim,
                entity_freq_threshold=self.hparams.entity_freq_threshold,
                entity_confidence_threshold=self.hparams.entity_confidence_threshold,
                train=False,
                validation=False,
                download=True
                )

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            trainset = MINDDataFrame(
                data_dir=self.hparams.data_dir, 
                size=self.hparams.size,
                mind_urls=self.hparams.mind_urls,
                word_embeddings_url=self.hparams.word_embeddings_url,
                word_embeddings_dirname=self.hparams.word_embeddings_dirname,
                word_embeddings_fpath=self.hparams.word_embeddings_fpath,
                entity_embeddings_filename=self.hparams.entity_embeddings_filename,
                id2index_filenames=self.hparams.id2index_filenames,
                dataset_attributes=self.hparams.dataset_attributes,
                word_embedding_dim=self.hparams.word_embedding_dim,
                entity_embedding_dim=self.hparams.entity_embedding_dim,
                entity_freq_threshold=self.hparams.entity_freq_threshold,
                entity_confidence_threshold=self.hparams.entity_confidence_threshold,
                train=True,
                validation=False,
                download=False
                )
            validset = MINDDataFrame(
                data_dir=self.hparams.data_dir, 
                size=self.hparams.size,
                mind_urls=self.hparams.mind_urls,
                word_embeddings_url=self.hparams.word_embeddings_url,
                word_embeddings_dirname=self.hparams.word_embeddings_dirname,
                word_embeddings_fpath=self.hparams.word_embeddings_fpath,
                entity_embeddings_filename=self.hparams.entity_embeddings_filename,
                id2index_filenames=self.hparams.id2index_filenames,
                dataset_attributes=self.hparams.dataset_attributes,
                word_embedding_dim=self.hparams.word_embedding_dim,
                entity_embedding_dim=self.hparams.entity_embedding_dim,
                entity_freq_threshold=self.hparams.entity_freq_threshold,
                entity_confidence_threshold=self.hparams.entity_confidence_threshold,
                train=True,
                validation=True,
                download=False
                )
            testset = MINDDataFrame(
                data_dir=self.hparams.data_dir, 
                size=self.hparams.size,
                mind_urls=self.hparams.mind_urls,
                word_embeddings_url=self.hparams.word_embeddings_url,
                word_embeddings_dirname=self.hparams.word_embeddings_dirname,
                word_embeddings_fpath=self.hparams.word_embeddings_fpath,
                entity_embeddings_filename=self.hparams.entity_embeddings_filename,
                id2index_filenames=self.hparams.id2index_filenames,
                dataset_attributes=self.hparams.dataset_attributes,
                word_embedding_dim=self.hparams.word_embedding_dim,
                entity_embedding_dim=self.hparams.entity_embedding_dim,
                entity_freq_threshold=self.hparams.entity_freq_threshold,
                entity_confidence_threshold=self.hparams.entity_confidence_threshold,
                train=False,
                validation=False,
                download=False
                )
        
            self.data_train = MINDDatasetTrain(
                    news=trainset.news,
                    behaviors=trainset.behaviors,
                    max_history_length=self.hparams.max_history_length,
                    neg_sampling_ratio=self.hparams.neg_sampling_ratio
                    )
            self.data_val = MINDDatasetTest(
                    news=validset.news,
                    behaviors=validset.behaviors,
                    max_history_length=self.hparams.max_history_length
                    )
            self.data_test = MINDDatasetTest(
                    news=testset.news,
                    behaviors=testset.behaviors,
                    max_history_length=self.hparams.max_history_length
                    )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            collate_fn=MINDCollate(self.hparams.max_title_length, self.hparams.max_abstract_length),
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            drop_last = self.hparams.drop_last
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            collate_fn=MINDCollate(self.hparams.max_title_length, self.hparams.max_abstract_length),
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            drop_last = self.hparams.drop_last
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            collate_fn=MINDCollate(self.hparams.max_title_length, self.hparams.max_abstract_length),
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            drop_last = self.hparams.drop_last
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils
    
    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "datamodule" / "mind.yaml")
    cfg.data_dir = str(root / "data")
    _ = hydra.utils.instantiate(cfg)
