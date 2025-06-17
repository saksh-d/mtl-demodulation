import torch
import torch.nn as nn

class CrossStitchUnit(nn.Module):
    def __init__(self):
        super().__init__()
        # Learnable 2x2 matrix initialized
        self.alpha = nn.Parameter(torch.tensor([[0.7, 0.3], [0.3, 0.7]], dtype=torch.float32))

    def forward(self, f1, f2):
        # Apply linear combination
        stacked = torch.stack([f1, f2], dim=0)  # [2, B, D]
        mixed = torch.einsum('ij,jbd->ibd', self.alpha, stacked)
        return mixed[0], mixed[1]

class CrossStitchMTL(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_phase_classes, num_pos_classes):
        super().__init__()
        reduced_dim = hidden_dim // 2

        # Lightweight per-task encoders
        self.enc_phase = nn.Sequential(
            nn.Linear(input_dim, reduced_dim),
            nn.ReLU()
        )
        self.enc_pos = nn.Sequential(
            nn.Linear(input_dim, reduced_dim),
            nn.ReLU()
        )

        self.cross_stitch = CrossStitchUnit()

        # Shared decoder head
        self.phase_head = nn.Sequential(
            nn.Linear(reduced_dim, reduced_dim),
            nn.ReLU(),
            nn.Linear(reduced_dim, num_phase_classes)
        )
        self.pos_head = nn.Sequential(
            nn.Linear(reduced_dim, reduced_dim),
            nn.ReLU(),
            nn.Linear(reduced_dim, num_pos_classes)
        )

    def forward(self, x):
        f1 = self.enc_phase(x)
        f2 = self.enc_pos(x)

        f1_cs, f2_cs = self.cross_stitch(f1, f2)

        return self.phase_head(f1_cs), self.pos_head(f2_cs)

