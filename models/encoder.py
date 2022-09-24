import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_layer = nn.Linear(WINDOW_SIZE * CHANNELS * BATCH_SIZE, 128)
        self.output_layer = nn.Linear(128, NUMBER_OF_AUGMENTATIONS)
        
    def forward(self, x):
        activation = self.hidden_layer(x)
        activation = torch.relu(activation)
        code = self.output_layer(activation)
        code = torch.relu(code)
        return code