from src.model import GGNNFlatSum, Devign
from src.metrics import BinarySensitivity
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics
import torch_geometric.data as geom_data
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau

architecture_dict = {'devign': Devign,
                     'flat': GGNNFlatSum}

def SavedPickle(path, file_to_save):
    with open(path, 'wb') as handle:
        pickle.dump(file_to_save, handle)

class DevignLightning(pl.LightningModule):
    """ Lightning module for training. Optimized with AdamW
    """
    def __init__(self, architecture_name: str, lr: float, **model_kwargs) -> None:
        """
        Args:
            architecture_name - Name of network architecture, key
                from [architecture_dict]
            model_kwargs - Additional arguments for the network
        """
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.model = architecture_dict[architecture_name](**model_kwargs)
        self.loss_module = nn.functional.binary_cross_entropy
        metrics = torchmetrics.MetricCollection([
            torchmetrics.F1Score(task='binary'),
            torchmetrics.Precision(task='binary'),
            torchmetrics.Recall(task='binary'),
            torchmetrics.Accuracy(task='binary', num_classes=2),
            torchmetrics.Specificity(task='binary'),
            BinarySensitivity(task='binary')
        ])
        self.train_metrics = metrics.clone(prefix='train_')
        self.test_metrics = metrics.clone(prefix='test_')
        self.val_metrics = metrics.clone(prefix='val_')
        self.feature_map_arr = []

    def forward(self, data: geom_data.batch.Batch) -> torch.Tensor:

        # input shape (data.x): torch.Size([25600, 115]) -- 115 is the embedding dimension.
        # input shape (data.edge_index): torch.Size([2, 12753])
        # input shape (data.batch): torch.Size([25600])
        """
        if return_attention:
            out_feature, mlp_feature, attention_data = self.model(data)
            return out_feature, mlp_feature, attention_data
        else:
        """
        x, feature_map = self.model(data)
        return x, feature_map

    def configure_optimizers(self) -> optim.Optimizer:
        optimizer = optim.AdamW(self.parameters(), lr = self.lr)
        return optimizer


    def training_step(self, 
                      batch: geom_data.batch.Batch,
                      batch_index: int) -> torch.Tensor:
        out, feature_map = self.forward(batch)
        y = batch.y
        loss = self.loss_module(out, y.float())
        preds = (out > 0.5).int()
        self.log("training_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=32)
        metric_out = self.train_metrics(preds, y)
        #self.log_dict(metric_out, on_step=True)
        #print("training_loss:" + str(loss))
        return loss
    
    def validation_step(self, 
                       batch: geom_data.batch.Batch,
                       batch_index: int) -> torch.Tensor:
        out, feature_map = self.forward(batch)
        preds = (out > 0.5).int()
        y = batch.y
        loss = self.loss_module(out, y.float())
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=32)
        metric_out = self.val_metrics(preds, batch.y)
        #self.log_dict(metric_out)
        #print("val_loss:" + str(loss))

    def test_step(self, 
                  batch: geom_data.batch.Batch,
                  batch_index: int) -> torch.Tensor:
        out, feature_map = self.forward(batch)
        preds = (out > 0.5).int()
        metric_out = self.test_metrics(preds, batch.y)
        #self.log_dict(metric_out)
        self.feature_map_arr.append(feature_map)
        #SavedPickle("feature_map.pkl", self.feature_map_arr)


