import torch.nn as nn
import torch

from .dfs_basic import Model as BasicModel


class BasicLayer(nn.Module):
    def __init__(self, n_in, n_out, actvation="ReLU", bias=False, p_dropout=0.1):
        super(BasicLayer, self).__init__()

        self.main = nn.Sequential(
            nn.Dropout(p=p_dropout),
            nn.Linear(n_in, n_out, bias=bias),
            nn.BatchNorm1d(n_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.main(x)

        return x


class ResidualLayer(nn.Module):
    def __init__(self, n_in, actvation="PReLU", bias=False, p_dropout=0.1):
        super(ResidualLayer, self).__init__()

        self.main = nn.Sequential(
            nn.Dropout(p=p_dropout), nn.Linear(n_in, n_in, bias=bias), nn.PReLU()
        )

    def forward(self, x):
        x = self.main(x) + x

        return x


class Model(BasicModel):
    def __init__(
        self,
        n_in,
        activation="PReLU",
        w_init=1,
        w_thresh=1e-2,
        n_mid=64,
        bias=False,
        p_dropout=0.1,
    ):
        super(Model, self).__init__()

        self.w = nn.Parameter(torch.zeros(n_in).float())
        nn.init.constant_(self.w, w_init)

        self.w_thresh = torch.Tensor([w_thresh])

        self.main = nn.Sequential(
            nn.Dropout(p=p_dropout),
            nn.Linear(n_in, n_mid, bias=bias),
            ResidualLayer(n_mid, p_dropout=p_dropout),
            nn.Dropout(p=p_dropout),
            nn.Linear(n_mid, n_in, bias=bias),
        )

        def weights_init(m):
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.2)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.normal_(m.weight, 0, 0.2)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.main.apply(weights_init)
