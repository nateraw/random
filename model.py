import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F


class DummyBackbone(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.act = nn.ReLU()

    def forward(self, x, y):
        x = x.view(x.size(0), -1)
        x = self.act(self.hidden(x))
        y_hat = self.act(self.fc(x))
        if y is None:
            return (y_hat,)
        loss = F.cross_entropy(y_hat, y)
        return (loss, y_hat)


class LitModel(pl.LightningModule):
    def __init__(self, c, h, w, num_classes, hidden_dim=128, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.model = DummyBackbone(
            input_dim=self.hparams.c * self.hparams.h * self.hparams.w,
            hidden_dim=self.hparams.hidden_dim,
            num_classes=self.hparams.num_classes,
        )
        self.example_input_array = (
            torch.randint(255, (32, self.hparams.c, self.hparams.h, self.hparams.w)).float(),
            torch.randint(self.hparams.num_classes, (32,)),
        )

    def forward(self, x, y):
        return self.model(x, y)

    def training_step(self, batch, batch_idx):
        loss, logits = self.model(*batch)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logits = self.model(*batch)
        self.log('val_loss', loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
