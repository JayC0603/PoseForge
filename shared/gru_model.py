"""Causal, single-layer GRU classifiers over the unchanged 52 pose features."""
from torch import nn


class PostureGRU(nn.Module):
    def __init__(self,input_dim=52,hidden_dim=64,dropout=.15):
        super().__init__()
        self.gru=nn.GRU(input_dim,hidden_dim,num_layers=1,batch_first=True,bidirectional=False)
        self.head=nn.Sequential(nn.Dropout(dropout),nn.Linear(hidden_dim,6))

    def forward(self,x):
        sequence,_=self.gru(x)
        return self.head(sequence[:,-1,:])
