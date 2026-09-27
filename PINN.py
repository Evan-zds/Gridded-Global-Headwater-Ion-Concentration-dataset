import torch
import torch.nn as nn
import pandas as pd
import torch.nn.functional as F
from PINN_Data import MyData
import math


def init_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight, gain=1.0)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


class MyModel(nn.Module):
    def __init__(self, input_dim, ndim, dropout, r_ff):
        super().__init__()
        self.input_dim = input_dim
        self.ndim = ndim

        self.input_head = nn.Sequential(
            nn.Linear(input_dim, self.ndim), nn.ReLU(), nn.Dropout(dropout), nn.BatchNorm1d(self.ndim),
        )
        self.backbone = nn.Sequential(
            nn.Linear(self.ndim, self.ndim * r_ff), nn.ReLU(), nn.Dropout(dropout), nn.BatchNorm1d(self.ndim * r_ff),
            nn.Linear(self.ndim * r_ff, self.ndim), nn.ReLU(), nn.Dropout(dropout), nn.BatchNorm1d(self.ndim),
        )
        self.output_head = nn.Sequential(
            nn.Linear(self.ndim, 128), nn.ReLU(), nn.Dropout(dropout), nn.BatchNorm1d(128),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(dropout), nn.BatchNorm1d(64),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(dropout), nn.BatchNorm1d(32),
            nn.Linear(32, 8),
            nn.Softplus()
        )
        self.ions = ['K', 'Na', 'Ca', 'Mg', 'Cl', 'SO4', 'NO3', 'HCO3']

    def forward(self, x):
        #  x: (input_dim)
        x_input_head = self.input_head(x)  # (ndim)
        x_backbone = self.backbone(x_input_head) + x_input_head  # (ndim)
        preds = self.output_head(x_backbone)
        x_output_head = {ion: preds[:, i].view(-1, 1) for i, ion in enumerate(self.ions)}
        return x_output_head

    def compute_loss(self, outputs, targets, lambda_cb):
        eps = 1e-5
        loss_data = 0
        delta_value = 1.0

        ion_weights = {'K': 1.0, 'Na': 1.0, 'Ca': 1.0, 'Mg': 1.0, 'Cl': 1.0, 'SO4': 1.0, 'NO3': 1.0, 'HCO3': 1.0}

        for ion in outputs:
            weight = ion_weights.get(ion, 1.0)
            log_pred = torch.log1p(outputs[ion] * 1000.0)
            log_true = torch.log1p(targets[ion] * 1000.0)
            loss_data += weight * F.huber_loss(log_pred, log_true, reduction='mean', delta=delta_value)

        cations = (outputs['K'] * 1 + outputs['Na'] * 1 + outputs['Ca'] * 2 + outputs['Mg'] * 2)
        anions = (outputs['HCO3'] * 1 + outputs['Cl'] * 1 + outputs['SO4'] * 2 + outputs['NO3'] * 1)
        cbe = (cations - anions) / (cations + anions + eps)
        cbe_penalty = torch.relu(torch.abs(cbe) - 0.1) ** 2
        loss_cb = torch.mean(cbe_penalty)
        total_loss = loss_data + lambda_cb * loss_cb
        return total_loss, loss_data, loss_cb