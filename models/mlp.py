import torch
import torch.nn as nn
        
class MLPModelLayernorm(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.input_dim = config.input_dim
        self.hidden_dims = config.hidden_dims
        self.out_dim = config.out_dim
        self.dropout = config.dropout
        self.num_layers = len(self.hidden_dims) + 1
        self.layers = nn.ModuleList()
        self.layers.append(nn.Sequential(nn.Linear(self.input_dim, self.hidden_dims[0]), 
                                         nn.LayerNorm(self.hidden_dims[0]),
                                         nn.ReLU(), 
                                         nn.Dropout(self.dropout)))
        for i in range(1, len(self.hidden_dims)):
            self.layers.append(nn.Sequential(nn.Linear(self.hidden_dims[i-1], self.hidden_dims[i]),
                                             nn.LayerNorm(self.hidden_dims[i]),
                                             nn.ReLU(),
                                             nn.Dropout(self.dropout)))
        self.layers.append(nn.Linear(self.hidden_dims[-1], self.out_dim))
        self.nc_only = False
        
    def forward(self, x):
        for i in range(self.num_layers - 1):
            x = self.layers[i](x)
        features = x
        if not self.nc_only:
            x = self.layers[-1](features)
        else:
            x = torch.zeros(x.shape[0], self.out_dim).to(x.device)
        return x, features
    
    def freeze_encoder(self):
        print('Freezing encoder (all layers except the last one)')
        for layer in self.layers[:-1]:
            for param in layer.parameters():
                param.requires_grad = False
    
    def enable_nc_only(self):
        self.nc_only = True
        print('Enable NC only mode')

if __name__ == '__main__':
    from easydict import EasyDict
    config = EasyDict({'input_dim': 1280, 'hidden_dims': [2560, 2000], 'out_dim': 1920, 'dropout': 0.5})
    
    model = MLPModelLayernorm(config)
    model.to('cuda:0')
    print(model)
    x = torch.randn(8, 1280).to('cuda:0')
    y, feat = model(x)
    print(y.shape, feat.shape)