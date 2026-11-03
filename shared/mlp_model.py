"""Minimal MLP architectures for the PoseForge 52-feature posture classifier."""

from torch import nn

MODEL_REGISTRY = {
    "mlp_small": (64, 32),
    "mlp_base": (128, 64),
    "mlp_wide": (256, 128),
}


class PostureMLP(nn.Module):
    def __init__(self, model_name="mlp_base", dropout=0.15, input_dim=52, num_classes=6):
        super().__init__()
        if model_name not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model: {model_name}; choose {list(MODEL_REGISTRY)}")
        h1, h2 = MODEL_REGISTRY[model_name]
        self.net = nn.Sequential(
            nn.Linear(input_dim, h1), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(h1, h2), nn.ReLU(),
            nn.Linear(h2, num_classes),
        )

    def forward(self, x):
        return self.net(x)


EXPERIMENTAL_MODEL_REGISTRY = ('mlp_residual', 'mlp_layernorm')


class ResidualBlock(nn.Module):
    def __init__(self, width, dropout):
        super().__init__()
        self.branch = nn.Sequential(nn.Linear(width,width), nn.ReLU(), nn.Dropout(dropout), nn.Linear(width,width))
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.activation(x + self.branch(x))


class ResidualPoseMLP(nn.Module):
    def __init__(self, input_dim=52, dropout=.15):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim,128), nn.ReLU(), nn.Dropout(dropout),
                                 ResidualBlock(128,dropout), ResidualBlock(128,dropout), nn.Linear(128,6))

    def forward(self, x):
        return self.net(x)


class LayerNormPoseMLP(nn.Module):
    def __init__(self, input_dim=52, dropout=.15):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim,256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(256,128), nn.LayerNorm(128), nn.ReLU(), nn.Linear(128,6))

    def forward(self, x):
        return self.net(x)


def build_classifier_model(model_name='mlp_base', dropout=.15, input_dim=52):
    if model_name in ('gru16','gru32'):
        from shared.gru_model import PostureGRU
        return PostureGRU(input_dim=input_dim,hidden_dim=64,dropout=dropout)
    if model_name in MODEL_REGISTRY:
        return PostureMLP(model_name, dropout, input_dim=input_dim)
    if model_name == 'mlp_residual':
        return ResidualPoseMLP(input_dim, dropout)
    if model_name == 'mlp_layernorm':
        return LayerNormPoseMLP(input_dim, dropout)
    raise ValueError(f'Unknown classifier architecture: {model_name}')
