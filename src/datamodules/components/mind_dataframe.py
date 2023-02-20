from ast import literal_eval
from typing import List, Dict, Tuple, Any, Union

import re
import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter
from torch.utils.data import Dataset

from src import utils
from src.datamodules.components.download_utils import maybe_download, download_path, extract_file

tqdm.pandas()

log = utils.get_pylogger(__name__)


class MINDDataFrame(Dataset):
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
            train: bool,
            validation: bool,
            download: bool
            ) -> None:

        super().__init__()

        self.data_dir = data_dir
        self.size = size
        self.mind_urls = mind_urls
        self.word_embeddings_url = word_embeddings_url

        self.word_embeddings_dirname = word_embeddings_dirname
        self.word_embeddings_fpath = word_embeddings_fpath
        self.entity_embeddings_filename = entity_embeddings_filename
        self.id2index_filenames = id2index_filenames
      
        self.dataset_attributes = dataset_attributes

        self.word_embedding_dim = word_embedding_dim
        self.entity_embedding_dim = entity_embedding_dim
        self.entity_freq_threshold = entity_freq_threshold
        self.entity_confidence_threshold = entity_confidence_threshold

        self.validation = validation
        
        if train:
            self.data_split = 'train'
        else:
            self.data_split = 'dev' # test

        self.dst_dir = os.path.join(
                self.data_dir,
                'MIND' + self.size + '_' + self.data_split
                )

        if download:
            self._download_and_extract()
            self._download_and_extract_embeddings()

        if not self._check_exists():
            raise RuntimeError('Dataset not found. Use download=True to download it.')

        self.news, self.behaviors = self.load_data()
    
    def __getitem__(self, idx: int) -> Tuple[Any, Any, Any]:
        user_bhv = self.behaviors.iloc[idx]

        history = user_bhv['history']
        candidates = user_bhv['candidates']
        labels = user_bhv['labels']

        history = self.news.loc[history]
        candidates = self.news.loc[candidates]
        labels = np.array(labels)

        return history, candidates, labels

    def __len__(self) -> int:
        return len(self.behaviors)

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """ Loads the parsed news and user behaviors.   

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]:
                Tuple of news and behaviors datasets.
        """
        news = self._load_news()
        log.info(f'News data size: {len(news)}')

        behaviors = self._load_behaviors()
        log.info(f'Behaviors data size for data split {self.data_split}, validation={self.validation}: {len(behaviors)}')

        return news, behaviors

    def _load_news(self):
        """ Loads the parsed news. If not already parsed, loads and preprocesses the raw news data.  

        Args:
            news (pd.DataFrame): Dataframe of news articles.

        Returns:
            pd.DataFrame: Parsed news data. 
        """
        file_suffix = '_all' if 'abstract' in self.dataset_attributes else ''
        parsed_news_file = os.path.join(self.dst_dir, 'parsed_news' + file_suffix + '.tsv')

        if self._check_integrity(parsed_news_file):
            # news data already parsed
            log.info(f'News data already parsed. Loading from {parsed_news_file}.')
            news = pd.read_table(
                    filepath_or_buffer=parsed_news_file,
                    converters={
                        attribute: literal_eval 
                        for attribute in ['title', 'abstract', 'title_entities', 'abstract_entities']
                        }
                    )

        else:
            log.info(f'News data not parsed. Loading and parsing raw data.')
            columns_names=['nid', 'category', 'subcategory', 'title', 'abstract', 'url', 'title_entities', 'abstract_entities']
            news = pd.read_table(
                    filepath_or_buffer=os.path.join(self.dst_dir, 'news.tsv'),
                    header=None,
                    names=columns_names,
                    usecols=range(len(columns_names))
                    )
            news = news.drop(columns=['url'])           
            
            # replace missing values
            news['abstract'].fillna('', inplace=True)
            news['title_entities'].fillna('[]', inplace=True)
            news['abstract_entities'].fillna('[]', inplace=True)
            
            # tokenize text
            news['title'] = news['title'].progress_apply(self.word_tokenize)
            news['abstract'] = news['abstract'].progress_apply(self.word_tokenize)

            entity_embeddings_fpath = os.path.join(self.dst_dir, self.entity_embeddings_filename)
            if 'abstract' in self.dataset_attributes:
                word2index_fpath = os.path.join(self.data_dir, 'MIND' + self.size + '_train', self.id2index_filenames['word2index_all'])
                entity2index_fpath = os.path.join(self.data_dir, 'MIND' + self.size + '_train', self.id2index_filenames['entity2index_all'])
                transformed_word_embeddings_filename = 'pretrained_word_embeddings_all'
                transformed_entity_embeddings_filename = 'pretrained_entity_embeddings_all'
            else:
                word2index_fpath = os.path.join(self.data_dir, 'MIND' + self.size + '_train', self.id2index_filenames['word2index'])
                entity2index_fpath = os.path.join(self.data_dir, 'MIND' + self.size + '_train', self.id2index_filenames['entity2index'])
                transformed_word_embeddings_filename = 'pretrained_word_embeddings'
                transformed_entity_embeddings_filename = 'pretrained_entity_embeddings'

            if self.data_split == 'train':
                # categ2index map
                log.info('Constructing categ2index map.')
                news_category = news['category'].drop_duplicates().reset_index(drop=True)
                categ2index = {v: k+1 for k, v in news_category.to_dict().items()}
                categ2index_fpath = os.path.join(self.dst_dir, self.id2index_filenames['categ2index'])
                log.info(f'Saving categ2index map of size {len(categ2index)} in {categ2index_fpath}')
                self._to_tsv(df=pd.DataFrame(categ2index.items(), columns=['category', 'index']),
                             fpath=categ2index_fpath)

                # subcateg2index map
                log.info('Constructing subcateg2index map.')
                news_subcategory = news['subcategory'].drop_duplicates().reset_index(drop=True)
                subcateg2index = {v: k+1 for k, v in news_subcategory.to_dict().items()}
                subcateg2index_fpath = os.path.join(self.dst_dir, self.id2index_filenames['subcateg2index'])
                log.info(f'Saving subcateg2index map of size {len(subcateg2index)} in {subcateg2index_fpath}')
                self._to_tsv(df=pd.DataFrame(subcateg2index.items(), columns=['subcategory', 'index']),
                             fpath=subcateg2index_fpath)

                # construct word2index map
                log.info('Constructing word2index map.')
                word_cnt = Counter() 
                for idx in tqdm(news.index.tolist()):
                    word_cnt.update(news.loc[idx]['title'])
                    if 'abstract' in self.dataset_attributes:
                        word_cnt.update(news.loc[idx]['abstract'])
                word2index = {k: v+1 for k, v in zip(word_cnt, range(len(word_cnt)))}
                log.info(f'Saving word2index map of size {len(word2index)} in {word2index_fpath}')
                self._to_tsv(df=pd.DataFrame(word2index.items(), columns=['word', 'index']),
                             fpath=word2index_fpath)
                    
                # construct word embedding matrix
                log.info('Constructing word embedding matrix.')
                self._generate_word_embeddings(
                        word2index = word2index,
                        embeddings_fpath = self.word_embeddings_fpath,
                        embedding_dim=self.word_embedding_dim,
                        transformed_embeddings_filename = transformed_word_embeddings_filename)

                log.info('Constructing entity2index map.')
                # keep only entities with a confidence over the threshold
                self.entity2freq = {}
                self._count_entity_freq(news['title_entities'])
                if 'abstract' in self.dataset_attributes:
                    self._count_entity_freq(news['abstract_entities'])

                # keep only entities with a frequency over the threshold
                self.entity2index = {}
                for entity, freq in self.entity2freq.items():
                    if freq > self.entity_freq_threshold:
                        self.entity2index[entity] = len(self.entity2index) + 1
                
                log.info(f'Saving entity2index map of size {len(self.entity2index)} in {entity2index_fpath}')
                self._to_tsv(
                        df=pd.DataFrame(self.entity2index.items(), columns=['entity', 'index']), 
                        fpath=entity2index_fpath
                        )
                
                # construct entity embedding matrix
                log.info('Constructing embedding embedding matrix.')
                self._generate_word_embeddings(
                        word2index = self.entity2index,
                        embeddings_fpath = entity_embeddings_fpath,
                        embedding_dim=self.entity_embedding_dim,
                        transformed_embeddings_filename = transformed_entity_embeddings_filename)

            else:
                log.info('Loading indices maps.')
                # load categ2index map
                categ2index_fpath = os.path.join(self.data_dir, 'MIND' + self.size + '_train', self.id2index_filenames['categ2index'])
                categ2index = self._load_idx_map_as_dict(categ2index_fpath)
               
                # load subcateg2index map
                subcateg2index_fpath = os.path.join(self.data_dir, 'MIND' + self.size + '_train', self.id2index_filenames['subcateg2index'])
                subcateg2index = self._load_idx_map_as_dict(subcateg2index_fpath)
               
                # load word2index map
                word2index = self._load_idx_map_as_dict(word2index_fpath)
               
                # load entity2index map
                self.entity2index = self._load_idx_map_as_dict(entity2index_fpath)

                # construct entity embedding matrix
                log.info('Constructing word embedding matrix.')
                self._generate_word_embeddings(
                        word2index = word2index,
                        embeddings_fpath = self.word_embeddings_fpath,
                        embedding_dim=self.word_embedding_dim,
                        transformed_embeddings_filename = transformed_word_embeddings_filename)

                # construct entity embedding matrix
                log.info('Constructing embedding embedding matrix.')
                self._generate_word_embeddings(
                        word2index = self.entity2index,
                        embeddings_fpath = entity_embeddings_fpath,
                        embedding_dim=self.entity_embedding_dim,
                        transformed_embeddings_filename = transformed_entity_embeddings_filename)

            # parse news
            log.info('Parsing news.')
            news['category'] = news['category'].progress_apply(lambda x: categ2index.get(x, 0))
            news['subcategory'] = news['subcategory'].progress_apply(lambda x: subcateg2index.get(x, 0))

            news['title'] = news['title'].progress_apply(lambda tokenized_title: [word2index.get(x, 0) for x in tokenized_title])
            news['abstract'] = news['abstract'].progress_apply(lambda tokenized_abstract: [word2index.get(x, 0) for x in tokenized_abstract])
            
            news['title_entities'] = news['title_entities'].progress_apply(lambda row: self._filter_entities(row))
            news['abstract_entities'] = news['abstract_entities'].progress_apply(lambda row: self._filter_entities(row))

            # cache parsed data
            log.info(f'Caching parsed news of size {len(news)} to {parsed_news_file}.')
            self._to_tsv(news, parsed_news_file)

        news = news.set_index('nid', drop=True)

        return news

    def _load_behaviors(self) -> pd.DataFrame:
        """ Loads the parsed user behaviors. If not already parsed, loads and parses the raw behavior data.  

        Returns:
            pd.DataFrame: Parsed user behavior data. 
        """
        file_prefix = ''
        if self.data_split == 'train':
            file_prefix = 'train_' if not self.validation else 'val_'
        parsed_behaviors_file = os.path.join(self.dst_dir, file_prefix + 'parsed_behaviors.tsv')

        if self._check_integrity(parsed_behaviors_file):
            # behaviors data already parsed
            log.info(f'User behaviors data already parsed. Loading from {parsed_behaviors_file}.')
            behaviors = pd.read_table(
                    filepath_or_buffer=parsed_behaviors_file,
                    converters={
                        'history': lambda x: x.strip("[]").replace("'","").split(", "),
                        'candidates': lambda x: x.strip("[]").replace("'","").split(", "),
                        'labels': lambda x: list(map(int, x.strip("[]").split(", "))),
                        }
                    )
        else:
            log.info(f'User behaviors data not parsed. Loading and parsing raw data.')
            columns_names=['impid', 'uid', 'time', 'history', 'impressions']
            behaviors = pd.read_table(
                    filepath_or_buffer=os.path.join(self.dst_dir, 'behaviors.tsv'),
                    header=None,
                    names=columns_names,
                    usecols=range(len(columns_names))
                    )

            # parse behaviors
            log.info('Parsing behaviors.')
            behaviors['time'] = pd.to_datetime(behaviors['time'], format='%m/%d/%Y %I:%M:%S %p')
            behaviors['history'] = behaviors['history'].fillna('').str.split()
            behaviors['impressions'] = behaviors['impressions'].str.split()
            behaviors['candidates'] = behaviors['impressions'].apply(
                    lambda x: [impression.split("-")[0] for impression in x ])
            behaviors['labels'] = behaviors['impressions'].apply(
                    lambda x: [int(impression.split("-")[1]) for impression in x ])
            behaviors = behaviors.drop(columns=['impressions'])

            # drop interactions of users without history 
            count_interactions = len(behaviors)
            behaviors = behaviors[behaviors['history'].apply(len) > 0]
            dropped_interactions = count_interactions - len(behaviors)
            log.info(f'Removed {dropped_interactions} ({dropped_interactions/count_interactions}%) interactions without user history.')
        
            behaviors = behaviors.reset_index(drop=True)

            if self.data_split == 'train':
                log.info('Splitting behavior data into train and validation sets.')
                if not self.validation:
                    # split behaviors into training dataset
                    behaviors = behaviors.loc[behaviors['time']<'2019-11-14 00:00:00']
                    behaviors = behaviors.reset_index(drop=True)

                    # compute uid2index map
                    log.info('Constructing uid2index map.')
                    uid2index = {}
                    for idx in tqdm(behaviors.index.tolist()):
                        uid = behaviors.loc[idx]['uid']
                        if uid not in uid2index:
                            uid2index[uid] = len(uid2index) + 1

                    fpath = os.path.join(self.dst_dir, self.id2index_filenames['uid2index'])
                    log.info(f'Saving uid2index map of size {len(uid2index)} in {fpath}')
                    self._to_tsv(df = pd.DataFrame(uid2index.items(), columns=['uid', 'index']),
                                 fpath = fpath)

                else:      
                    # split behaviors into validation dataset
                    behaviors = behaviors.loc[behaviors['time']>='2019-11-14 00:00:00']
                    behaviors = behaviors.reset_index(drop=True)

                    # load uid2index map
                    log.info('Loading uid2index map.')
                    fpath = os.path.join(self.data_dir, 'MIND' + self.size + '_train', self.id2index_filenames['uid2index'])
                    uid2index = self._load_idx_map_as_dict(fpath)
           
            else:
                # load uid2index map
                log.info('Loading uid2index map.')
                fpath = os.path.join(self.data_dir, 'MIND' + self.size + '_train', self.id2index_filenames['uid2index'])
                uid2index = self._load_idx_map_as_dict(fpath)
           
            # map uid to index
            log.info('Mapping uid to index.')
            behaviors['user'] = behaviors['uid'].apply(lambda x: uid2index.get(x, 0))
            behaviors = behaviors[['user', 'history', 'candidates', 'labels']]

            # cache processed data
            log.info(f'Caching parsed behaviors of size {len(behaviors)} to {parsed_behaviors_file}.')
            self._to_tsv(behaviors, parsed_behaviors_file)

        return behaviors

    def word_tokenize(self, sentence: str) -> List[str]:
        """Splits a sentence into word list using regex.

        Args:
            sentence (str): input sentence

        Returns:
            list: word list
        """
        pat = re.compile(r"[\w]+|[.,!?;|]")
        if isinstance(sentence, str):
            return pat.findall(sentence.lower())
        else:
            return []
    
    def _generate_word_embeddings(self, word2index: Dict[str, int], embeddings_fpath: str, embedding_dim: int, transformed_embeddings_filename: Union[str, None]) -> None:
        """ Loads pretrained embeddings for the words (or entities) in word_dict.

        Args:
            word2index (Dict[str, int]): word dictionary
            embeddings_fpath (str): the filepath of the embeddings to be loaded
            ebedding_dim (int): dimensionality of embeddings
            transformed_embeddings_filename (str): the name of the transformed embeddings file
        """

        embedding_matrix = np.random.normal(size=(len(word2index) + 1, embedding_dim))
        exist_word = set()

        with open(embeddings_fpath, "r") as f:
            for line in tqdm(f):
                linesplit = line.split(" ")
                word = line[0]
                if len(word) != 0:
                    if word in word2index:
                        embedding_matrix[word2index[word]] = np.asarray(list(map(float, linesplit[1:])))
                        exist_word.add(word)
        
        log.info(f'Rate of word missed in pretrained embedding: {(len(exist_word)/len(word2index))}.')

        fpath = os.path.join(self.dst_dir, transformed_embeddings_filename)
        if not self._check_integrity(fpath):
            log.info(f'Saving word embeddings in {fpath}')
            np.save(fpath, embedding_matrix, allow_pickle=True)

    def _count_entity_freq(self, data: pd.Series) -> None:
        for row in tqdm(data):
            for entity in json.loads(row):
                times = len(entity['OccurrenceOffsets']) * entity['Confidence']
                if times > 0:
                    if entity['WikidataId'] not in self.entity2freq:
                        self.entity2freq[entity['WikidataId']] = times
                    else:
                        self.entity2freq[entity['WikidataId']] += times

    def _filter_entities(self, data: pd.Series) -> List[int]:
        filtered_entities = []
        for entity in json.loads(data):
            if entity['Confidence'] > self.entity_confidence_threshold and entity['WikidataId'] in self.entity2index:
                filtered_entities.append(self.entity2index[entity['WikidataId']])
        return filtered_entities

    def _download_and_extract(self) -> None:
        """ Downloads the MIND dataset in the specified size, if not already downloaded, then extracts it. """
       
        # download the dataset
        url = self.mind_urls[self.size][self.data_split] 
        log.info(f'Downloading MIND {self.size} dataset for {self.data_split} from {url}.')
        
        with download_path(self.data_dir) as path:
            path = maybe_download(
                    url=url,
                    filename=url.split('/')[-1],
                    work_directory=path
                    )
            log.info(f'Compressed dataset downloaded.')

            # extract the compressed data files
            log.info(f'Extracting dataset from {path} into {self.dst_dir}.')
            extract_file(
                    archive_file=path,
                    dst_dir=self.dst_dir,
                    clean_archive=False
                    )
            log.info(f'Dataset extraction completed.')

    def _download_and_extract_embeddings(self) -> None:
        """ Downloads and extracts Glove embeddings, if not already downloaded."""
        log.info(f"Downloading Glove embeddings from {self.word_embeddings_url}.")

        glove_dst_dir = os.path.join(
                self.data_dir, self.word_embeddings_dirname
                )

        # download the embeddings
        with download_path(self.data_dir) as path:
            path = maybe_download(
                    url=self.word_embeddings_url,
                    filename=self.word_embeddings_url.split('/')[-1],
                    work_directory=path
                    )
            log.info(f'Compressed Glove embeddings downloaded.')

            # extract the compressed file
            if not self._check_integrity(self.word_embeddings_fpath):
                log.info(f'Extracting Glove embeddings from {path} intp {glove_dst_dir}.')
                extract_file(
                        archive_file=path,
                        dst_dir=glove_dst_dir,
                        clean_archive=False
                        )
                log.info(f'Embeddings extraction completed.')

    def _to_tsv(self, df: pd.DataFrame, fpath: str) -> None:
        df.to_csv(fpath, sep='\t', index=False)

    def _load_idx_map_as_dict(self, fpath: str) -> Dict[str, int]:
        idx_map_dict = dict(pd.read_table(fpath).values.tolist())
        return idx_map_dict

    def _check_exists(self) -> bool:
        return os.path.isdir(self.dst_dir) and os.listdir(self.dst_dir)

    def _check_integrity(self, fpath: str) -> bool:
        if not os.path.isfile(fpath):
            return False
        return True

