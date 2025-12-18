import torch
import torch.nn as nn
import torch.nn.functional as F


class BranchNet(nn.Module):
    def __init__(self, input_dim, hidden_channels, output_dim, kernel_size=3):
        super(BranchNet, self).__init__()
        # Assuming input is (batch_size, input_dim)
        # We reshape to (batch_size, 1, input_dim) for Conv1d

        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=hidden_channels[0],
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )
        self.conv2 = nn.Conv1d(
            in_channels=hidden_channels[0],
            out_channels=hidden_channels[1],
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )

        # Calculate flattened size after convolutions
        # (assuming padding keeps length same)
        self.flatten_dim = hidden_channels[1] * input_dim

        self.fc1 = nn.Linear(self.flatten_dim, 128)
        self.fc2 = nn.Linear(128, output_dim)

    def forward(self, x):
        # x shape: [batch_size, input_dim]
        x = x.unsqueeze(1)  # [batch_size, 1, input_dim]

        x = F.tanh(self.conv1(x))
        x = F.tanh(self.conv2(x))

        x = x.view(x.size(0), -1)  # Flatten
        x = F.tanh(self.fc1(x))
        x = self.fc2(x)
        return x


class TrunkNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3):
        super(TrunkNet, self).__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.Tanh())  # Tanh is commonly used in PINNs/DeepONets

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())

        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x shape: [num_points, 2]
        return self.net(x)


class DeepONet(nn.Module):
    def __init__(
        self,
        branch_input_dim,
        trunk_input_dim,
        p,
        branch_channels=None,
        trunk_hidden_dim=128,
        default_trunk_input = None
    ):
        """
        Args:
            branch_input_dim: Dimension of latent variables
                              (e.g., speed, length, time)
            trunk_input_dim: Dimension of coordinates (2 for 2D grid)
            p: Number of basis functions (output dimension of branch and trunk)
            branch_channels: List of output channels for CNN layers in branch
            trunk_hidden_dim: Hidden dimension for MLP in trunk
        """
        super(DeepONet, self).__init__()

        if branch_channels is None:
            branch_channels = [16, 32]

        self.branch = BranchNet(branch_input_dim, branch_channels, p)
        self.trunk = TrunkNet(trunk_input_dim, trunk_hidden_dim, p)
        self.bias = nn.Parameter(torch.zeros(1))
        
        if default_trunk_input is not None:
            self.default_trunk_input = default_trunk_input
        else:
            self.default_trunk_input = None
        
    def forward(self, u, y=None):
        """
        Args:
            u: Branch input (latent variables) [batch_size, branch_input_dim]
            y: Trunk input (grid coordinates) [num_points, 2]

        Returns:
            Output field [batch_size, num_points]
        """
        if y is None:
            y = self.default_trunk_input

        # Branch output: [batch_size, p]
        b_out = self.branch(u)

        # Trunk output: [num_points, p]
        t_out = self.trunk(y)

        # Dot product: (B, P) x (N, P)^T -> (B, N)
        # where N = nx * ny
        output = torch.matmul(b_out, t_out.t()) + self.bias

        return output


# Example usage
if __name__ == "__main__":
    # Hyperparameters
    batch_size = 10
    latent_dim = 5  # e.g., speed, relative length, time, etc.
    nx, ny = 20, 20
    num_points = nx * ny
    p = 50  # Number of basis functions

    model = DeepONet(branch_input_dim=latent_dim, trunk_input_dim=2, p=p)

    # Dummy inputs
    u = torch.randn(batch_size, latent_dim)  # Latent variables
    y = torch.randn(num_points, 2)          # Grid points

    # Forward pass
    output = model(u, y)

    print(f"Input u shape: {u.shape}")
    print(f"Input y shape: {y.shape}")
    print(f"Output shape: {output.shape}")  # Should be [10, 400]
