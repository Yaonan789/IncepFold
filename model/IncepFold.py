import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class Inception1D(nn.Module):
    def __init__(self,  in_channels, out_channels):
        super(Inception1D, self).__init__()
        branch_channels = out_channels // 4

        # Branch 1:  1x1 conv
        self.branch1 = nn.Sequential(
            nn.Conv1d(in_channels, branch_channels, kernel_size=1),
            nn.BatchNorm1d(branch_channels),
            nn.LeakyReLU()
        )

        # Branch 2: 1x1 conv -> 3x3 conv
        self.branch2 = nn.Sequential(
            nn.Conv1d(in_channels, branch_channels, kernel_size=1),
            nn.BatchNorm1d(branch_channels),
            nn.LeakyReLU(),
            nn.Conv1d(branch_channels, branch_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(branch_channels),
            nn.LeakyReLU()
        )

        # Branch 3: 1x1 conv -> 3x3 conv -> 5x5 conv
        self.branch3 = nn.Sequential(
            nn.Conv1d(in_channels, branch_channels, kernel_size=1),
            nn.BatchNorm1d(branch_channels),
            nn.LeakyReLU(),
            nn.Conv1d(branch_channels, branch_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(branch_channels),
            nn.LeakyReLU(),
            nn.Conv1d(branch_channels, branch_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(branch_channels),
            nn.LeakyReLU()
        )

        # Branch 4: MaxPool -> 1x1 conv
        self.branch4 = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, branch_channels, kernel_size=1),
            nn.BatchNorm1d(branch_channels),
            nn.LeakyReLU()
        )

        self.residual = nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, kernel_size=1),
                    nn.BatchNorm1d(out_channels)
                )
    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)
        out = torch.cat([x1,x2,x3,x4], dim=1)
        res = self.residual(x)
        out = F.leaky_relu(out + res)
        return out


class Encoder(nn.Module):
    def __init__(self, in_dim, output_dim, base_channels=32,  num_layers = 12, num_bins=256):
        super(Encoder, self).__init__()
        self.initial = nn.Sequential(
            nn.Conv1d(in_dim, base_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(base_channels),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )

        self.layers = nn.ModuleList()
        current_channels = base_channels

        
        channel_multipliers = [1, 1, 2, 2, 4, 4, 8, 8, 16, 16, 32, 32][:num_layers]

        for i, mult in enumerate(channel_multipliers):
            out_channels = min(output_dim, base_channels * mult)

            #Inception
            self.layers.append(
                Inception1D(current_channels, out_channels)
            )

            self.layers.append(
                nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
            )

            current_channels = out_channels

    def forward(self, x):
        x = x.permute(0, 2, 1)  # [B, L, C] -> [B, C, L]
        x = self.initial(x)
        for layer in self.layers:
            x = layer(x)
        return x



class ResBlockDilated(nn.Module):
    def __init__(self, size, hidden = 64, stride = 1, dil = 2):
        super(ResBlockDilated, self).__init__()
        pad_len = dil 
        self.res = nn.Sequential(
                        nn.Conv2d(hidden, hidden, size, padding = pad_len, 
                            dilation = dil),
                        nn.BatchNorm2d(hidden),
                        nn.ReLU(),
                        nn.Conv2d(hidden, hidden, size, padding = pad_len,
                            dilation = dil),
                        nn.BatchNorm2d(hidden),
                        )
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x 
        res_out = self.res(x)
        out = self.relu(res_out + identity)
        return out


class Decoder(nn.Module):
    def __init__(self, in_channel, hidden = 256, filter_size = 3, num_blocks = 5):
        super(Decoder, self).__init__()
        self.filter_size = filter_size

        self.conv_start = nn.Sequential(
                                    nn.Conv2d(in_channel, hidden, 3, 1, 1),
                                    nn.BatchNorm2d(hidden),
                                    nn.ReLU(),
                                    )
        self.res_blocks = self.get_res_blocks(num_blocks, hidden)
        self.conv_end = nn.Conv2d(hidden, 1, 1)

    def forward(self, x):
        x = self.conv_start(x)
        x = self.res_blocks(x)
        out = self.conv_end(x)
        return out

    def get_res_blocks(self, n, hidden):
        blocks = []
        for i in range(n):
            dilation = 2 ** (i + 1)
            blocks.append(ResBlockDilated(self.filter_size, hidden = hidden, dil = dilation))
        res_blocks = nn.Sequential(*blocks)
        return res_blocks


class IncepFold(nn.Module):
    def __init__(self, in_dim=6, output_size=256):
        super(IncepFold, self).__init__()
        self.encoder = Encoder(in_dim, output_size)
        self.decoder = Decoder(2*output_size)

    def forward(self, x):
        x = self.encoder(x)
        x = self.diagonalize(x)
        x = self.decoder(x).squeeze(1) 
        return x
    
    def diagonalize(self, x):
        x_i = x.unsqueeze(2).repeat(1, 1, 256, 1)
        x_j = x.unsqueeze(3).repeat(1, 1, 1, 256)
        input_map = torch.cat([x_i, x_j], dim = 1)
        return input_map



def print_output_shape(module, input, output):

    if isinstance(output, tuple):
        print(f"Module: {module.__class__.__name__}, Output is a tuple with {len(output)} elements")
        for i, item in enumerate(output):
            if hasattr(item, 'shape'):
                print(f"  Element {i}: Shape {item.shape}")
            else:
                print(f"  Element {i}: Not a tensor")
    else:
        print(f"Module: {module.__class__.__name__}, Output Shape: {output.shape}")


def register_hooks(model):
    hooks = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Sequential) and not isinstance(module, nn.ModuleList):
            hook = module.register_forward_hook(print_output_shape)
            hooks.append(hook)
    return hooks


if __name__ == '__main__':
    input = torch.rand(1, 2097152, 6)
    print(f'input.shape:{input.shape}')
    model = IncepFold()
    hooks = register_hooks(model)
    output = model(input)
    print(f'out.shape:{output.shape}')
