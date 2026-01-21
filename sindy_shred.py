import torch
from torch.utils.data import DataLoader
import numpy as np
from sindy import sindy_library_torch, e_sindy_library_torch

class E_SINDy(torch.nn.Module):
    def __init__(self, num_replicates, latent_dim, library_dim, poly_order, include_sine, device='cpu'):
        super().__init__()
        self.num_replicates = num_replicates
        self.latent_dim = latent_dim
        self.poly_order = poly_order
        self.include_sine = include_sine
        self.library_dim = library_dim
        self.device = device
        self.coefficients = torch.ones(num_replicates, library_dim, latent_dim, requires_grad=True)
        torch.nn.init.normal_(self.coefficients, mean=0.0, std=0.001)
        self.coefficient_mask = torch.ones(num_replicates, library_dim, latent_dim, requires_grad=False).to(self.device)
        self.coefficients = torch.nn.Parameter(self.coefficients)

    def forward(self, h_replicates, dt):
        num_data, num_replicates, latent_dim = h_replicates.shape
        h_replicates = h_replicates.reshape(num_data * num_replicates, latent_dim)
        library_Thetas = e_sindy_library_torch(h_replicates, self.latent_dim, self.poly_order, self.include_sine)
        library_Thetas = library_Thetas.reshape(num_data, num_replicates, self.library_dim)
        h_replicates = h_replicates.reshape(num_data, num_replicates, latent_dim)
        h_replicates = h_replicates + torch.einsum('ijk,jkl->ijl', library_Thetas, (self.coefficients * self.coefficient_mask)) * dt
        # print(" SINDy Coefficients: ", self.coefficients.data)
        return h_replicates
    
    def thresholding(self, threshold, base_threshold=0):
        threshold_tensor = torch.full_like(self.coefficients, threshold)
        for i in range(self.num_replicates):
            threshold_tensor[i] = threshold_tensor[i] * 10**(0.2 * i - 1) + base_threshold
        self.coefficient_mask = torch.abs(self.coefficients) > threshold_tensor
        self.coefficients.data = self.coefficient_mask * self.coefficients.data
        
class MLP(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_size=64, hidden_layers=2, dropout=0.0):
        super(MLP, self).__init__()
        layers = []
        layers.append(torch.nn.Linear(input_size, hidden_size))
        layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Dropout(dropout))
        for _ in range(hidden_layers - 1):
            layers.append(torch.nn.Linear(hidden_size, hidden_size))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(dropout))
        layers.append(torch.nn.Linear(hidden_size, output_size))
        self.network = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

from DeepONet import DeepONet
from ShiftDeepOnet_single_basis import ShiftDeepONet

class SINDy_SHRED(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_size=64, hidden_layers=1, l1=350, l2=400, dropout=0.0, library_dim=10, poly_order=3, include_sine=False, dt=0.03, device='cpu', use_layer_norm=False, use_DON=False, use_SDON = False, nx = None, ny = None, P =50):
        # hidden size is the latent dimension
        
        # Robustly handle scalar or (Nx, Ny) output_size
        if isinstance(output_size, int) or np.isscalar(output_size) or (torch.is_tensor(output_size) and output_size.dim() == 0):
            # Accept int, numpy scalar, or 0-dim tensor
            self.output_size = int(output_size)
        else:
            # output size is a list/tuple: Nx, Ny
            self.Nx = int(output_size[0])
            self.Ny = int(output_size[1])
            # if there are more entries, throw a warning
            if len(output_size) > 2:
                print("Warning: output_size has more than 2 entries, only the first two will be used.")
                self.output_size = output_size.item()
            
        
        super(SINDy_SHRED, self).__init__()
        self.device = device
        self.gru = torch.nn.GRU(input_size=input_size, hidden_size=hidden_size,
                                        num_layers=hidden_layers, batch_first=True).to(device)
        self.num_replicates = 5
        self.num_euler_steps = 3
        self.e_sindy = E_SINDy(self.num_replicates, hidden_size, library_dim, poly_order, include_sine, device=device)
        
        self.linear1 = torch.nn.Linear(hidden_size, l1)
        self.linear2 = torch.nn.Linear(l1, l2)
        self.linear3 = torch.nn.Linear(l2, self.output_size)

        self.dropout = torch.nn.Dropout(dropout)

        self.hidden_layers = hidden_layers
        self.hidden_size = hidden_size
        self.dt = dt
        self.use_layer_norm = use_layer_norm

        self.layer_norm = torch.nn.LayerNorm(hidden_size).to(self.device)
        self.to(device)
        self.use_DON = use_DON
        
        # Initialize the 2D DeepONet architecture if use_DON is True
        
        if self.use_DON or use_SDON:
            if nx is None or ny is None:
                raise ValueError("nx and ny must be provided when use_DON is True.")
            branch_dim = hidden_size
            
                # Grid generation
            x = torch.linspace(0, 1, nx)
            y = torch.linspace(0, 1, ny)
            grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')

            # Trunk input: [nx*ny, 2]
            flat_x = grid_x.reshape(-1)
            flat_y = grid_y.reshape(-1)
            self.trunk_inputs = torch.stack([flat_x, flat_y], dim=1).to(device)
            if use_SDON:
                print(" ShiftDeepONet architecture selected - to be implemented")
                print("SDON architecture with a single change of coordinates A(u), b(u) for all basis functions.")
                self.deepONet =  ShiftDeepONet(
                    branch_input_dim=branch_dim,
                    trunk_input_dim=2,
                    p=P,
                    shift_scale_channels=[l1, l2]
                ).to(device)
            else:   
                print("DON architecture selected")
                self.deepONet =  DeepONet(
                    branch_input_dim=branch_dim,
                    trunk_input_dim=2,
                    p=P,
                    branch_channels=[l1, l2],
                    trunk_hidden_dim=[l1, l2]
                ).to(device)
            
        
        
    def forward(self, x, sindy=False):
        h_0 = torch.zeros((self.hidden_layers, x.size(0), self.hidden_size), dtype=torch.float).to(self.device)
        
        out, h_out = self.gru(x, h_0)
        
         # Apply layer normalization if enabled
        if self.use_layer_norm:
            # Normalize per time-step across hidden dimension
            out = self.layer_norm(out)
            # Normalize the final hidden state from last layer
            h_last = self.layer_norm(h_out[-1])
        else:
            h_last = h_out[-1]

        h_out = h_last.view(-1, self.hidden_size)
        # print("GRU output shape: - what we pass to the MLP", h_out.shape)
        if self.use_DON:
            # print("Using DON architecture")
            # TODO: implement DON architecture
            branch_inputs = h_out
            output = self.deepONet( branch_inputs,  self.trunk_inputs)
            # print("Final output shape of the DeepONet:", output.shape ) # should be [batch_size, nx*ny]
        else:
            output = self.linear1(h_out)
            output = self.dropout(output)
            output = torch.nn.functional.relu(output)

            output = self.linear2(output)
            output = self.dropout(output)
            output = torch.nn.functional.relu(output)
        
            output = self.linear3(output)
            # print("Final output shape:", output.shape )

        with torch.autograd.set_detect_anomaly(True):
            if sindy:
                h_t = h_out[:-1, :]
                ht_replicates = h_t.unsqueeze(1).repeat(1, self.num_replicates, 1)
                for _ in range(self.num_euler_steps):
                    ht_replicates = self.e_sindy(ht_replicates, dt=self.dt/float(self.num_euler_steps))
                h_out_replicates = h_out[1:, :].unsqueeze(1).repeat(1, self.num_replicates, 1)
                output = output, h_out_replicates, ht_replicates
        return output
    
    def gru_outputs(self, x, sindy=False):
        h_0 = torch.zeros((self.hidden_layers, x.size(0), self.hidden_size), dtype=torch.float).to(self.device)
        _, h_out = self.gru(x, h_0)
        h_out = h_out[-1].view(-1, self.hidden_size)

        if sindy:
            h_t = h_out[:-1, :]
            ht_replicates = h_t.unsqueeze(1).repeat(1, self.num_replicates, 1)
            for _ in range(self.num_euler_steps):
                ht_replicates = self.e_sindy(ht_replicates, dt=self.dt/float(self.num_euler_steps))
            h_out_replicates = h_out[1:, :].unsqueeze(1).repeat(1, self.num_replicates, 1)
            h_outs = h_out_replicates, ht_replicates
        return h_outs
    
    def sindys_threshold(self, threshold):
        self.e_sindy.thresholding(threshold)
            

def fit(model, train_dataset, valid_dataset, batch_size=64, num_epochs=4000, lr=1e-3, sindy_regularization=1.0, mean_zero_regularization = 0.5, variance_regularization = 0.5, background_regularization = 0.5, optimizer="AdamW", verbose=False, threshold=0.5, base_threshold=0.0, patience=20, thres_epoch=100, weight_decay=0.01):
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size) # shufffle should be false!!
    criterion = torch.nn.MSELoss()
    # Attempt to compile the model with torch.compile (PyTorch 2.0+).
    # Compile before creating the optimizer so the optimizer binds to the compiled model's parameters.
    model = torch.compile(model)
    if optimizer == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    val_error_list = []
    patience_counter = 0
    best_params = model.state_dict()
    for epoch in range(1, num_epochs + 1):
        for data in train_loader:
            model.train()
            outputs, h_gru, h_sindy = model(data[0], sindy=True)
            # print(" shapes in training loop - outputs, h_gru, h_sindy: ", outputs.shape, h_gru.shape, h_sindy.shape)
            optimizer.zero_grad()
            # print("Output shape:", outputs.shape, " Target shape:", data[1].shape)
            loss = criterion(outputs, data[1]) + criterion(h_gru, h_sindy) * sindy_regularization  + torch.abs(torch.mean(h_gru)) * mean_zero_regularization + torch.var(h_gru) * variance_regularization + background_regularization * torch.norm(h_gru[1:, :, 0] - h_gru[:-1, :, 0])
            loss.backward()
            optimizer.step()
        print(epoch, ":", loss)
        if epoch % thres_epoch == 0 and epoch != 0:
            model.e_sindy.thresholding(threshold=threshold, base_threshold=base_threshold)
            model.eval()
            with torch.no_grad():
                val_outputs = model(valid_dataset.X)
                val_error = torch.linalg.norm(val_outputs - valid_dataset.Y)
                val_error = val_error / torch.linalg.norm(valid_dataset.Y)
                val_error_list.append(val_error)
            if verbose:
                print('Training epoch ' + str(epoch))
                print('Error ' + str(val_error_list[-1]))
            if val_error == torch.min(torch.tensor(val_error_list)):
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter == patience:
                return torch.tensor(val_error_list).cpu()
    return torch.tensor(val_error_list).detach().cpu().numpy()

def forecast(forecaster, reconstructor, test_dataset):
    initial_in = test_dataset.X[0:1].clone()
    vals = [initial_in[0, i, :].detach().cpu().clone().numpy() for i in range(test_dataset.X.shape[1])]
    for i in range(len(test_dataset.X)):
        scaled_output1, scaled_output2 = forecaster(initial_in)
        scaled_output1 = scaled_output1.detach().cpu().numpy()
        scaled_output2 = scaled_output2.detach().cpu().numpy()
        vals.append(np.concatenate([scaled_output1.reshape(test_dataset.X.shape[2]//2), scaled_output2.reshape(test_dataset.X.shape[2]//2)]))
        temp = initial_in.clone()
        initial_in[0, :-1] = temp[0, 1:]
        initial_in[0, -1] = torch.tensor(np.concatenate([scaled_output1, scaled_output2]))
    device = 'cuda' if next(reconstructor.parameters()).is_cuda else 'cpu'
    forecasted_vals = torch.tensor(np.array(vals), dtype=torch.float32).to(device)
    reconstructions = []
    for i in range(len(forecasted_vals) - test_dataset.X.shape[1]):
        recon = reconstructor(forecasted_vals[i:i + test_dataset.X.shape[1]].reshape(1, test_dataset.X.shape[1], test_dataset.X.shape[2])).detach().cpu().numpy()
        reconstructions.append(recon)
    reconstructions = np.array(reconstructions)
    return forecasted_vals, reconstructions
