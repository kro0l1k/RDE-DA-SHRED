#ShiftDeepOnet_single_basis.py
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

class BranchNet_mlp(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3):
        super(BranchNet_mlp, self).__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.Tanh())  # Tanh is commonly used in PINNs/DeepONets

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())

        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x shape: [batch_size, input_dim]
        return self.net(x)

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


class ShiftDeepONet(nn.Module):
    """
    DeepONet with an input-dependent affine transform of trunk coordinates:
        y' = y @ A(u)^T + b(u)
    where A(u) ∈ R^{d×d}, b(u) ∈ R^{d}, d = trunk_input_dim.
    """
    def __init__(
        self,
        branch_input_dim: int,
        trunk_input_dim: int,
        p: int,
        branch_channels=None,
        trunk_hidden_dim: int = 128,
        shift_scale_channels=None,
    ):
        super().__init__()

        if branch_channels is None:
            branch_channels = [8, 16]
            
        if shift_scale_channels is None:
            shift_scale_channels = [8, 8]
            
        self.p = p
        self.d = trunk_input_dim

        self.branch = BranchNet(branch_input_dim, branch_channels, p)
        self.trunk  = TrunkNet(trunk_input_dim, trunk_hidden_dim, p)

        
        # NOTE: we are using a single transformation instead of per-basis:
        print("[ShiftDeepONet]: Using single affine transform A(u), b(u) for all basis functions.")
        # Output an affine map in coordinate space (NOT per-basis):
        # A(u): d*d parameters, b(u): d parameters
        print("using MLP for shift and scale networks.")
        
        # Extract hidden dimension if it's a list
        hidden_dim = shift_scale_channels[0] if isinstance(shift_scale_channels, list) else shift_scale_channels
        
        self.scale = BranchNet_mlp(branch_input_dim, hidden_dim, trunk_input_dim * trunk_input_dim)
        self.shift = BranchNet_mlp(branch_input_dim, hidden_dim, trunk_input_dim)

        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, u: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        u: [B, m]
        y: [N, d]
        returns: [B, N]
        """
        B = u.shape[0]
        N = y.shape[0]
        d = y.shape[1]
        assert d == self.d, f"Expected y.shape[1]=={self.d}, got {d}"

        y = y.to(device=u.device, dtype=u.dtype)

        # Branch basis coefficients: [B, p]
        b_out = self.branch(u)

        # Affine transform parameters
        A = self.scale(u).view(B, d, d)     # [B, d, d]
        b = self.shift(u).view(B, 1, d)     # [B, 1, d]

        # Apply y' = y @ A^T + b
        # y: [N, d], A: [B, d, d] -> y_t: [B, N, d]
        y_t = torch.einsum("nd,bdk->bnk", y, A) + b

        # Trunk basis evals: [B, N, p]
        t_out = self.trunk(y_t.reshape(B * N, d)).view(B, N, self.p)

        # DeepONet combine: sum_k b_k(u) * t_k(y')
        out = torch.einsum("bp,bnp->bn", b_out, t_out) + self.bias
        return out



# Example usage
if __name__ == "__main__":
    # Hyperparameters
    batch_size = 10
    latent_dim = 5  # e.g., speed, relative length, time, etc.
    nx, ny = 20, 20
    num_points = nx * ny
    p = 50  # Number of basis functions

    model = ShiftDeepONet(branch_input_dim=latent_dim, trunk_input_dim=2, p=p)

    # Dummy inputs
    u = torch.randn(batch_size, latent_dim)  # Latent variables
    y = torch.randn(num_points, 2)          # Grid points

    # Forward pass
    output = model(u, y)

    print(f"Input u shape: {u.shape}")
    print(f"Input y shape: {y.shape}")
    print(f"Output shape: {output.shape}")  # Should be [10, 400]
