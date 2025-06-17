import torch
import torch.nn as nn

class SplitMTL(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_phase_classes, num_pos_classes):
        super().__init__()

        self.phase_branch = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_phase_classes)
        )

        self.pos_branch = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_pos_classes)
        )

    def forward(self, x):
        phase_out = self.phase_branch(x)
        pos_out = self.pos_branch(x)
        return phase_out, pos_out
