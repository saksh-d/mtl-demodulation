import torch
import torch.nn as nn

class SharedMTL(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_phase_classes, num_pos_classes):
        super().__init__()
        self.shared_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.phase_head = nn.Linear(hidden_dim, num_phase_classes)
        self.pos_head = nn.Linear(hidden_dim, num_pos_classes)

    def forward(self, x):
        shared = self.shared_layers(x)
        phase_out = self.phase_head(shared)
        pos_out = self.pos_head(shared)
        return phase_out, pos_out
