import torch
import torch.nn as nn

class AveragedDenoiser(nn.Module):
    def __init__(self, R, beta):
        super(AveragedDenoiser, self).__init__()
        self.R = R  # The given neural network
        assert 0 < beta < 1, "Beta should be in the range (0, 1)"
        self.beta = nn.Parameter(torch.tensor(beta), requires_grad=False)  # The coefficient beta (should be a value between 0 and 1)
    
    def forward(self, x):
        identity = x  # The identity operation (Id)
        R_output = self.R(x)  # Apply the neural network R to the input x
        return self.beta * R_output + (1 - self.beta) * identity
