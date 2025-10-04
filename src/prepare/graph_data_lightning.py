
from src.prepare import DatasetBuilder, JoernExeError
from src.prepare import JoernExeError
import os
import pytorch_lightning as pl
import torch
import torch.utils.data as data_utils
import torch_geometric.data as geom_data
import torch_geometric.transforms as geom_transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from src.utils import load_pickle

class GraphDataLightning(pl.LightningDataModule):
    """ Lightning Data Module for graph datasets. Batch size 128.
    """
    def __init__(self,
                 fresh_build: bool,       
                 test_build: bool, 
                 num_nodes: int,
                 train_proportion: float,
                 batch_size: int) -> None:
        """
        Args:
            fresh_build: True to generate dataset source files and graphs,
                deleting any existing. False to reuse existing files.
            test_build: True to use a subset of the data 
                (from the project directory's test folder), 
                False if all data is to be used.
            num_nodes: The number of nodes to be used in an individual CPG.
            train_proportion: Proportion of samples used for the training dataset.
                The rest is split in half for test/val.
            batch_size

        Raises:
            JoernExeError: If running Joern fails
        """
        super().__init__()
        try:
            # dataset is loaded initially.
            # dataset, dataset_id = DatasetBuilder(fresh_build, test_build).build_graphs(num_nodes) # Uncomment this for the first time.
        # Load the saved dataset
            #dataset = load_pickle("dataset_cpg_multi-edge_codebert_fine-tuned.pkl")
            #dataset_id  = load_pickle("dataset_cpg_multi-edge_codebert_fine-tuned_id.pkl")

            train_set = load_pickle("sard_cwe_word2vec_train.pkl")
            test_set = load_pickle("sard_cwe_word2vec_test.pkl")
            validation_set = load_pickle("sard_cwe_word2vec_val.pkl")
            train_set_id = load_pickle("sard_cwe_word2vec_train_id.pkl")
            test_set_id = load_pickle("sard_cwe_word2vec_test_id.pkl")
            validation_set_id = load_pickle("sard_cwe_word2vec_val_id.pkl")

        except JoernExeError:
            raise JoernExeError
        # split the dataset into training, validation, and test sets.

        train_y = []
        test_y = []
        validation_y = []
        for item in train_set:
            train_y.append(int(item.y))
        for item in test_set:
            test_y.append(int(item.y))
        for item in validation_set:
            validation_y.append(int(item.y))
        """
        data_targets = []
        for item in dataset:
            data_targets.append(int(item.y))
        train_indices, val_test_indices, train_y, val_test_y = train_test_split(range(len(dataset)), data_targets, stratify=data_targets, test_size = 0.4, random_state= 42)
        train_set = Subset(dataset, train_indices)
        val_test_set = Subset(dataset, val_test_indices)
        train_set_id = Subset(dataset_id, train_indices)
        val_test_set_id = Subset(dataset_id, val_test_indices)

        validation_indices, test_indices, validation_y, test_y = train_test_split(range(len(val_test_set)), val_test_y, stratify=val_test_y, test_size = 0.5, random_state= 42)
        validation_set = Subset(val_test_set, validation_indices)
        test_set = Subset(val_test_set, test_indices)
        validation_set_id = Subset(val_test_set_id, validation_indices)
        test_set_id = Subset(val_test_set_id, test_indices)

        #train_count = int(len(dataset) * train_proportion)
        #holdout_count = len(dataset) - train_count
        #test_count = int(holdout_count/2)
        #val_count = holdout_count - test_count
        #split_counts = [train_count, test_count, val_count]
        #self.train_set, self.test_set, self.val_set = data_utils.random_split(dataset, split_counts)
        """
        self.batch_size = batch_size

        self.train_set = train_set
        self.test_set= test_set
        self.val_set = validation_set

        self.train_set_id = train_set_id
        self.validation_set_id = validation_set_id
        self.test_set_id = test_set_id

        self.train_y = train_y
        self.validation_y = validation_y
        self.test_y = test_y


    def train_dataloader(self) -> geom_data.DataLoader:
        negative_sample_cnt = len([x for x in self.train_set if x.y == 0])
        positive_sample_cnt = len([x for x in self.train_set if x.y == 1])
        # more negative samples than positive in this dataset

        oversample_rate = negative_sample_cnt / positive_sample_cnt
        class_weights = [1, oversample_rate]
        sample_weights = [0] * len(self.train_set)
        #for index, (_, _, _, y) in enumerate(self.train_set):
        for index, (_, _, y) in enumerate(self.train_set):
            label = y[1][0].item()
            class_weight = class_weights[label]
            sample_weights[index] = class_weight
        sampler = data_utils.WeightedRandomSampler(sample_weights,
                                                   num_samples=len(self.train_set))
        loader = geom_data.DataLoader(self.train_set,
                                      batch_size = self.batch_size,
                                      #num_workers = os.cpu_count(),
                                      num_workers= 0,
                                      sampler = sampler)
        return loader
    
    def test_dataloader(self) -> geom_data.DataLoader:
        loader = geom_data.DataLoader(self.test_set,
                                      batch_size= self.batch_size,
                                      num_workers=0)
                                      #num_workers=os.cpu_count())
        return loader

    def val_dataloader(self) -> geom_data.DataLoader:
        loader = geom_data.DataLoader(self.val_set,
                                      batch_size= self.batch_size,
                                      num_workers=0)
                                      #num_workers=os.cpu_count())
        return loader

    def sample_dataloader(self) -> geom_data.DataLoader:
        loader = geom_data.DataLoader(self.train_set,
                                      batch_size=self.batch_size,
                                      num_workers=0)
                                      # num_workers=os.cpu_count())
        return loader

    def train_dataloader_no_sampler(self) -> geom_data.DataLoader:
        """用于评估的原始训练集（无采样）"""
        return geom_data.DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            num_workers=0,
            shuffle=False  # 保持原始顺序
        )
