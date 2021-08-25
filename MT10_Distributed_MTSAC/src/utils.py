import json
import torch.nn as nn


class Decoder(json.JSONDecoder):
    def decode(self, s):
        result = super().decode(s)  
        return self._decode(result)

    def _decode(self, o):
        if isinstance(o, str):
            try:
                return int(o)
            except ValueError:
                return o
        elif isinstance(o, dict):
            return {k: self._decode(v) for k, v in o.items()}
        elif isinstance(o, list):
            return [self._decode(v) for v in o]
        else:
            return o


def cfg_read(path):
    with open(path, 'r') as f:
        cfg = json.loads(f.read(), cls=Decoder)
    return cfg


def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=1)
        nn.init.constant_(m.bias, 0)


def build_mlp(input_dim, output_dim, hidden_dims):
    '''
    Not include actiavtion of output layer
    '''
    network = nn.ModuleList()
    dims = [input_dim] + hidden_dims
    for in_dim_, out_dim_ in zip(dims[:-1], dims[1:]):
        network.append(
            nn.Linear(
                in_features=in_dim_,
                out_features=out_dim_
            )
        )
        network.append(nn.ReLU())
    network.append(
        nn.Linear(
            in_features=hidden_dims[-1],
            out_features=output_dim
        )
    )

    return nn.Sequential(*network)
