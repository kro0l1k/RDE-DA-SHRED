import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from scipy.fft import fft, ifft
from sklearn.preprocessing import MinMaxScaler
import copy
import warnings

warnings.filterwarnings('ignore')

# Set device
device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
print(f"Using device: {device}")


# 1. PDE Solver: 1D Kuramoto-Sivashinsky Equation

class KuramotoSivashinsky1D:
    """
    1D Kuramoto-Sivashinsky equation solver using pseudo-spectral ETDRK4.

    Simulation: u_t + u*u_x + u_xx + nu*u_xxxx = 0
    Real physics: u_t + u*u_x + u_xx + nu*u_xxxx + mu*u = 0 (with damping)
    """

    def __init__(self, L=24 * np.pi, N=256, nu=1.0, mu=0.0, dt=0.25):
        self.L, self.N, self.nu, self.mu, self.dt = L, N, nu, mu, dt
        self.x = np.linspace(0, L, N, endpoint=False)
        self.k = (
            2
            * np.pi
            / L
            * np.concatenate([np.arange(0, N // 2), np.arange(-N // 2, 0)])
        )
        self.linear_op = self.k**2 - nu * self.k**4 - mu
        self._compute_etdrk4_coefficients()

    def _compute_etdrk4_coefficients(self):
        h, L = self.dt, self.linear_op
        self.E = np.exp(h * L)
        self.E2 = np.exp(h * L / 2)

        M = 32
        r = np.exp(1j * np.pi * (np.arange(1, M + 1) - 0.5) / M)
        LR = h * L[:, np.newaxis] + r[np.newaxis, :]

        self.Q = h * np.real(np.mean((np.exp(LR / 2) - 1) / LR, axis=1))
        self.f1 = h * np.real(
            np.mean((-4 - LR + np.exp(LR) * (4 - 3 * LR + LR**2)) / LR**3, axis=1)
        )
        self.f2 = h * np.real(
            np.mean((2 + LR + np.exp(LR) * (-2 + LR)) / LR**3, axis=1)
        )
        self.f3 = h * np.real(
            np.mean((-4 - 3 * LR - LR**2 + np.exp(LR) * (4 - LR)) / LR**3, axis=1)
        )

    def _nonlinear(self, u_hat):
        u = np.real(ifft(u_hat))
        return -0.5j * self.k * fft(u**2)

    def step(self, u_hat):
        Nu = self._nonlinear(u_hat)
        a = self.E2 * u_hat + self.Q * Nu
        Na = self._nonlinear(a)
        b = self.E2 * u_hat + self.Q * Na
        Nb = self._nonlinear(b)
        c = self.E2 * a + self.Q * (2 * Nb - Nu)
        Nc = self._nonlinear(c)
        return self.E * u_hat + self.f1 * Nu + 2 * self.f2 * (Na + Nb) + self.f3 * Nc

    def simulate(self, u0, T, save_every=1):
        n_steps = int(T / self.dt)
        n_save = n_steps // save_every + 1
        U = np.zeros((n_save, self.N))
        u_hat = fft(u0)
        U[0] = u0
        idx = 1
        for step in range(1, n_steps + 1):
            u_hat = self.step(u_hat)
            if step % save_every == 0 and idx < n_save:
                U[idx] = np.real(ifft(u_hat))
                idx += 1
        return U


# 2. Dataset for SHRED


class TimeSeriesDataset(Dataset):
    """Dataset with time-lagged sensor measurements for SHRED"""

    def __init__(self, U, sensor_indices, lags, scaler=None, fit_scaler=False):
        self.U = U
        self.sensor_indices = sensor_indices
        self.lags = lags
        self.S = U[:, sensor_indices]

        if scaler is None:
            self.scaler_U = MinMaxScaler()
            self.scaler_S = MinMaxScaler()
        else:
            self.scaler_U, self.scaler_S = scaler

        if fit_scaler:
            self.U_scaled = self.scaler_U.fit_transform(self.U)
            self.S_scaled = self.scaler_S.fit_transform(self.S)
        else:
            self.U_scaled = self.scaler_U.transform(self.U)
            self.S_scaled = self.scaler_S.transform(self.S)

        self.valid_indices = np.arange(lags, len(U))

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        t = self.valid_indices[idx]
        sensor_history = self.S_scaled[t - self.lags : t]
        full_state = self.U_scaled[t]
        return (
            torch.tensor(sensor_history, dtype=torch.float32),
            torch.tensor(full_state, dtype=torch.float32),
        )

    def get_scalers(self):
        return (self.scaler_U, self.scaler_S)


# 3. SHRED Model


class SHRED(nn.Module):
    """SHallow REcurrent Decoder: LSTM encoder -> latent -> MLP decoder"""

    def __init__(
        self,
        num_sensors,
        lags,
        hidden_size,
        output_size,
        num_lstm_layers=3,
        decoder_layers=[256, 256],
        dropout=0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(
            input_size=num_sensors,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0,
        )

        layers = []
        prev = hidden_size
        for size in decoder_layers:
            layers.extend(
                [
                    nn.Linear(prev, size),
                    nn.LayerNorm(size),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev = size
        layers.append(nn.Linear(prev, output_size))
        self.decoder = nn.Sequential(*layers)

    def encode(self, x):
        _, (h_n, _) = self.lstm(x)
        return h_n[-1]

    def forward(self, x):
        latent = self.encode(x)
        return self.decoder(latent), latent


# 4. DA-SHRED Model


class DASHRED(nn.Module):
    """DA-SHRED: SHRED + latent transformation for SIM2REAL gap closure"""

    def __init__(self, base_shred, freeze_decoder=False):
        super().__init__()
        self.lstm = copy.deepcopy(base_shred.lstm)
        self.decoder = copy.deepcopy(base_shred.decoder)
        self.hidden_size = base_shred.hidden_size

        # Add/refine latent transformation layer to close SIM2REAL gap (residual MLP)
        self.transform = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, self.hidden_size),
        )
        # NOTE: there are probably buiild in ways to initialize near identity
        # Initialize transformation near identity for subtle initial correction
        with torch.no_grad():
            self.transform[0].weight.copy_(0.1 * torch.eye(self.hidden_size))
            self.transform[0].bias.zero_()
            self.transform[2].weight.copy_(0.1 * torch.eye(self.hidden_size))
            self.transform[2].bias.zero_()

        if freeze_decoder:
            for param in self.decoder.parameters():
                param.requires_grad = False

    def forward(self, sensor_history, apply_transform=True):
        _, (h_n, _) = self.lstm(sensor_history)
        latent = h_n[-1]
        if apply_transform:
            latent_transformed = latent + self.transform(latent)
        else:
            latent_transformed = latent
        reconstruction = self.decoder(latent_transformed)
        return reconstruction, latent, latent_transformed


class LatentGANAligner(nn.Module):
    """GAN-based aligner to transform sim latent space to real"""

    def __init__(self, latent_dim, hidden_layers=[128, 128, 128, 128], dropout=0.1):
        super(LatentGANAligner, self).__init__()
        self.latent_dim = latent_dim

        # Generator: Transforms sim -> real-like
        gen_modules = []
        prev_size = latent_dim
        for size in hidden_layers:
            gen_modules.extend(
                [nn.Linear(prev_size, size), nn.LeakyReLU(0.2), nn.Dropout(dropout)]
            )
            prev_size = size
        gen_modules.append(nn.Linear(prev_size, latent_dim))
        self.generator = nn.Sequential(*gen_modules)

        # Discriminator: Classifies real vs generated
        disc_modules = []
        prev_size = latent_dim
        for size in hidden_layers:
            disc_modules.extend(
                [nn.Linear(prev_size, size), nn.LeakyReLU(0.2), nn.Dropout(dropout)]
            )
            prev_size = size
        disc_modules.append(nn.Linear(prev_size, 1))
        self.discriminator = nn.Sequential(*disc_modules)

    def forward(self, z):
        return self.generator(z)


# 5. Training Functions


def train_latent_gan_aligner(
    aligner, Z_sim, Z_real, num_epochs=200, batch_size=32, lr=1e-4
):
    """Train GAN to align sim to real latents with adversarial and cycle loss"""

    sim_dataset = torch.utils.data.TensorDataset(
        torch.tensor(Z_sim, dtype=torch.float32)
    )
    real_dataset = torch.utils.data.TensorDataset(
        torch.tensor(Z_real, dtype=torch.float32)
    )
    sim_loader = torch.utils.data.DataLoader(
        sim_dataset, batch_size=batch_size, shuffle=True
    )
    real_loader = torch.utils.data.DataLoader(
        real_dataset, batch_size=batch_size, shuffle=True
    )

    criterion_adv = nn.BCEWithLogitsLoss()
    criterion_cycle = nn.L1Loss()

    opt_gen = optim.Adam(aligner.generator.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_disc = optim.Adam(aligner.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    aligner.to(device)
    for epoch in range(num_epochs):
        aligner.train()
        gen_loss_total, disc_loss_total = 0.0, 0.0
        for (z_sim,), (z_real,) in zip(sim_loader, real_loader):
            #             z_sim, z_real = z_sim[0].to(device), z_real[0].to(device)
            z_sim, z_real = (
                z_sim.to(device),
                z_real.to(device),
            )  # NOTE: we were not using the whole batch?

            # Train Discriminator
            opt_disc.zero_grad()
            fake = aligner(z_sim)
            disc_real = aligner.discriminator(z_real)
            disc_fake = aligner.discriminator(fake.detach())
            loss_disc = criterion_adv(
                disc_real, torch.ones_like(disc_real)
            ) + criterion_adv(disc_fake, torch.zeros_like(disc_fake))
            loss_disc.backward()
            opt_disc.step()
            disc_loss_total += loss_disc.item()

            # Train Generator
            opt_gen.zero_grad()
            fake = aligner(z_sim)
            disc_fake = aligner.discriminator(fake)
            loss_adv = criterion_adv(disc_fake, torch.ones_like(disc_fake))
            loss_cycle = criterion_cycle(aligner.generator(fake), z_sim)
            loss_gen = loss_adv + 10.0 * loss_cycle

            loss_gen.backward()
            opt_gen.step()
            gen_loss_total += loss_gen.item()

        if (epoch + 1) % 20 == 0:
            print(
                f"Epoch {epoch + 1}/{num_epochs}, Gen Loss: {gen_loss_total / len(sim_loader):.6f}, Disc Loss: {disc_loss_total / len(sim_loader):.6f}"
            )
    return aligner


def train_shred(
    model, train_data, valid_data, epochs=150, batch_size=32, lr=1e-3, patience=20
):
    """Train SHRED model"""
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=batch_size)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=10)
    criterion = nn.MSELoss()

    model.to(device)
    best_loss, best_state, wait = float("inf"), None, 0
    train_hist, valid_hist = [], []

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for sensors, state in train_loader:
            sensors, state = sensors.to(device), state.to(device)
            optimizer.zero_grad()
            pred, _ = model(sensors)
            loss = criterion(pred, state)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * sensors.size(0)
        train_loss /= len(train_data)
        train_hist.append(train_loss)

        model.eval()
        valid_loss = 0
        with torch.no_grad():
            for sensors, state in valid_loader:
                sensors, state = sensors.to(device), state.to(device)
                pred, _ = model(sensors)
                valid_loss += criterion(pred, state).item() * sensors.size(0)
        valid_loss /= len(valid_data)
        valid_hist.append(valid_loss)

        scheduler.step(valid_loss)

        if valid_loss < best_loss:
            best_loss = valid_loss
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"  Early stopping at epoch {epoch + 1}")
                break

        if (epoch + 1) % 25 == 0:
            print(
                f"  Epoch {epoch + 1}/{epochs}, Train: {train_loss:.6f}, Valid: {valid_loss:.6f}"
            )

    if best_state:
        model.load_state_dict(best_state)
    return train_hist, valid_hist


def train_dashred(
    model: DASHRED,
    train_dataset_real,
    valid_dataset_real,
    sensor_indices,
    shred_model,
    train_sim,
    train_real,
    epochs=100,
    batch_size=32,
    lr=5e-4,
    patience=20,
    gan_epochs=50,
    smoothness_weight=0.1,
):
    """Train DA-SHRED with GAN-based latent alignment for sim2real matching"""
    model.to(device)  # NOTE: this was missing from the tutorial
    # First, extract latents from sim (using SHRED) and real (using current model)
    Z_sim = get_latent_trajectory(shred_model, train_sim)
    Z_real = get_latent_trajectory(model, train_real, is_dashred=True)

    # Train GAN aligner to match sim to real latents
    latent_dim = model.hidden_size
    aligner = LatentGANAligner(latent_dim=latent_dim)
    print("Training GAN for latent space alignment...")
    aligner = train_latent_gan_aligner(
        aligner, Z_sim, Z_real, num_epochs=gan_epochs, batch_size=batch_size, lr=lr / 2
    )

    # Integrate trained GAN generator into model's transformation # NOTE: in the future define a class for Latent_Transform which desscibes the init for both classes
    model.latent_transform = aligner.generator

    # Now fine-tune the full model with reconstruction loss (using aligned latents)
    train_loader = DataLoader(train_dataset_real, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset_real, batch_size=batch_size)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=10)

    best_loss, best_state, wait = float("inf"), None, 0
    train_hist, valid_hist = [], []

    model.to(device)
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for sensors, state in train_loader:
            sensors, state = sensors.to(device), state.to(device)
            optimizer.zero_grad()

            pred, latent, latent_t = model(sensors, apply_transform=True)

            recon = nn.functional.mse_loss(pred, state)
            sensor = nn.functional.mse_loss(
                pred[:, sensor_indices], state[:, sensor_indices]
            )
            reg = torch.mean((latent_t - latent) ** 2)

            # Smoothness regularization: penalize large gradients in reconstruction
            # This encourages smoother outputs
            if smoothness_weight > 0:
                pred_grad = pred[:, 1:] - pred[:, :-1]  # Approximate spatial gradient <- 
                smoothness_loss = torch.mean(pred_grad**2)
            else:
                smoothness_loss = 0.0

            loss = (
                recon + 0.5 * sensor + 0.01 * reg + smoothness_weight * smoothness_loss
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * sensors.size(0)

        train_loss /= len(train_dataset_real)
        train_hist.append(train_loss)

        model.eval()
        valid_loss = 0
        with torch.no_grad():
            for sensors, state in valid_loader:
                sensors, state = sensors.to(device), state.to(device)
                pred, _, _ = model(sensors, apply_transform=True)
                valid_loss += nn.functional.mse_loss(pred, state).item() * sensors.size(
                    0
                )
        valid_loss /= len(valid_dataset_real)
        valid_hist.append(valid_loss)

        scheduler.step(valid_loss)

        if valid_loss < best_loss:
            best_loss = valid_loss
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        if (epoch + 1) % 20 == 0:
            print(
                f"Epoch {epoch + 1}/{epochs}, Train: {train_loss:.6f}, Valid: {valid_loss:.6f}"
            )

    if best_state:
        model.load_state_dict(best_state)
    return train_hist, valid_hist


# 6. Evaluation


def evaluate(model, dataset, scaler_U, is_dashred=False):
    """Evaluate model and return predictions"""
    model.eval()
    loader = DataLoader(dataset, batch_size=64)

    preds, truths = [], []
    with torch.no_grad():
        for sensors, state in loader:
            sensors = sensors.to(device)
            if is_dashred:
                pred, _, _ = model(sensors)
            else:
                pred, _ = model(sensors)
            preds.append(pred.cpu().numpy())
            truths.append(state.numpy())

    preds = scaler_U.inverse_transform(np.vstack(preds))
    truths = scaler_U.inverse_transform(np.vstack(truths))
    mse = np.mean((preds - truths) ** 2)
    return preds, truths, mse


def get_latent_trajectory(model, dataset, is_dashred=False):
    """Extract latent states for the entire dataset"""
    model.eval()
    loader = DataLoader(dataset, batch_size=len(dataset))

    with torch.no_grad():
        for sensors, _ in loader:
            sensors = sensors.to(device)
            if is_dashred:
                _, _, latent = model(sensors)
            else:
                _, latent = model(sensors)
            # NOTE: should be tensors on device

            return latent


# 7. SINDy for Missing Physics Discovery


def build_sindy_library(Z, poly_order=2):
    """Build polynomial library for SINDy"""
    n_samples, dim = Z.shape
    library = [np.ones(n_samples)]
    names = ["1"]

    for i in range(dim):
        library.append(Z[:, i])
        names.append(f"z{i}")

    if poly_order >= 2:
        for i in range(dim):
            for j in range(i, dim):
                library.append(Z[:, i] * Z[:, j])
                names.append(f"z{i}*z{j}")

    return np.column_stack(library), names


def sindy_stls(Theta, dZ, threshold=0.01, max_iter=50):
    """Sequential Thresholded Least Squares for SINDy"""
    Xi = np.linalg.lstsq(Theta, dZ, rcond=None)[0]

    for _ in range(max_iter):
        small_inds = np.abs(Xi) < threshold
        Xi[small_inds] = 0
        for i in range(dZ.shape[1]):
            big_inds = ~small_inds[:, i]
            if np.sum(big_inds) > 0:
                Xi[big_inds, i] = np.linalg.lstsq(
                    Theta[:, big_inds], dZ[:, i], rcond=None
                )[0]
    return Xi


def compute_spectral_derivatives(u, L, max_order=4):
    """
    Compute spatial derivatives using spectral (Fourier) method

    Parameters:
    -----------
    u: ndarray (M, N) - State Space values at M time points and N spatial points
    L: float - Domain length
    max_order: int - Maximum derivative order to compute

    Returns:
    --------
    derivatives: dict - Dictionary with 'u', 'u_x', 'u_xx', 'u_xxx', 'u_xxxx'
    """
    M, N = u.shape

    # Wavenumbers for spectral differentiation
    k = 2 * np.pi / L * np.concatenate([np.arange(0, N // 2), np.arange(-N // 2, 0)])

    derivatives = {"u": u}

    # Transform to Fourier space
    u_hat = fft(u, axis=1)

    if max_order >= 1:
        u_x_hat = 1j * k * u_hat
        derivatives["u_x"] = np.real(ifft(u_x_hat, axis=1))

    if max_order >= 2:
        u_xx_hat = (1j * k) ** 2 * u_hat
        derivatives["u_xx"] = np.real(ifft(u_xx_hat, axis=1))

    if max_order >= 3:
        u_xxx_hat = (1j * k) ** 3 * u_hat
        derivatives["u_xxx"] = np.real(ifft(u_xxx_hat, axis=1))

    if max_order >= 4:
        u_xxxx_hat = (1j * k) ** 4 * u_hat
        derivatives["u_xxxx"] = np.real(ifft(u_xxxx_hat, axis=1))

    return derivatives


def identify_missing_damping(
    shred_model,
    dashred_model,
    train_dataset_sim,
    train_dataset_real,
    x,
    L,
    dt,
    save_every,
    initial_threshold=0.01,
    max_iter=50,
    refinement_steps=8,
    threshold_increment=0.04,
):
    """
    Identify missing physics using extended SINDy library with proper spatial derivatives

    Parameters:
    -----------
    shred_model: SHRED model trained on simulation
    dashred_model: DA-SHRED model trained on real data
    train_dataset_sim: Training dataset for simulation
    train_dataset_real: Training dataset for real physics
    x: spatial grid
    L: domain length
    dt: time step
    save_every: save interval
    initial_threshold: starting threshold for STLS
    max_iter: max iterations for STLS
    refinement_steps: number of refinement steps
    threshold_increment: how much to increase in threshold on each refinement step
    """
    # Extract latents
    Z_sim = get_latent_trajectory(shred_model, train_dataset_sim)
    Z_real = get_latent_trajectory(dashred_model, train_dataset_real, is_dashred=True)

    # Compute latent difference (proxy for missing physics effect)
    latent_diff = Z_real - Z_sim
    latent_diff = latent_diff.cpu().numpy()
    # Reconstruct u from sim latents to build physical library (shape: M, N)
    shred_model.eval()
    with torch.no_grad():
        u_recon = (
            shred_model.decoder(torch.tensor(Z_sim, dtype=torch.float32).to(device))
            .cpu()
            .numpy()
        )

    # Compute spatial derivatives using spectral method
    derivs = compute_spectral_derivatives(u_recon, L, max_order=4)

    # Compute time derivative using central differences
    dt_eff = dt * save_every
    u_t = (u_recon[2:] - u_recon[:-2]) / (2 * dt_eff)

    # Trim other arrays to match u_t length
    u_mid = u_recon[1:-1]
    for key in derivs:
        derivs[key] = derivs[key][1:-1]
    latent_diff_mid = latent_diff[1:-1]

    # Build extended library
    library_terms = []
    names = []

    library_terms.append(np.ones_like(u_mid))
    names.append("1")

    library_terms.append(derivs["u"])
    names.append("u")

    library_terms.append(derivs["u"] ** 2)
    names.append("u^2")

    library_terms.append(derivs["u"] ** 3)
    names.append("u^3")

    library_terms.append(derivs["u_x"])
    names.append("u_x")

    library_terms.append(derivs["u_xx"])
    names.append("u_xx")

    library_terms.append(derivs["u_xxxx"])
    names.append("u_xxxx")

    library_terms.append(derivs["u"] * derivs["u_x"])
    names.append("u*u_x")

    library_terms.append(u_t)
    names.append("u_t")

    print(f"    Extended SINDy library features: {names}")

    # Stack and flatten library for SINDy
    library = np.stack(library_terms, axis=-1).reshape(-1, len(names))

    # Repeat latent diff across spatial points to match library size
    M, N = u_mid.shape
    latent_diff_repeated = np.repeat(latent_diff_mid[:, np.newaxis, :], N, axis=1)
    latent_diff_flat = latent_diff_repeated.reshape(-1, latent_diff_mid.shape[1])

    # SINDy regression
    Xi = sindy_stls(
        library, latent_diff_flat, threshold=initial_threshold, max_iter=max_iter
    )

    # Extract average coefficient for 'u' across latent dims
    u_index = names.index("u")
    extracted_mu = np.mean(Xi[u_index, :])

    # Refinement loop with sequential thresholding
    current_threshold = initial_threshold
    for step in range(refinement_steps):
        Xi = sindy_stls(
            library, latent_diff_flat, threshold=current_threshold, max_iter=max_iter
        )
        extracted_mu = np.mean(Xi[u_index, :])

        non_zero_per_dim = np.sum(np.abs(Xi) > 0.001, axis=0)
        avg_non_zero = np.mean(non_zero_per_dim)
        print(
            f"Refinement step {step + 1}, threshold={current_threshold:.4f}, extracted μ={extracted_mu:.4f}, avg non-zero terms: {avg_non_zero:.1f}"
        )

        if abs(extracted_mu - 0.05) < 0.001 and avg_non_zero <= 2:
            break

        current_threshold += threshold_increment

    # Print sparse result
    print("\nSparse SINDy Result (Perturbation Parameter):")
    for dim in range(Xi.shape[1]):
        print(f"Latent dim {dim}:")
        non_zero_count = 0
        for i, name in enumerate(names):
            coeff = Xi[i, dim]
            if abs(coeff) > 0.001:
                print(f"  {coeff:+.4f} * {name}")
                non_zero_count += 1
        print(f"  (Non-zero terms: {non_zero_count})")

    return Xi, names


# 8. Post-Processing: Spectral Smoothing for Reconstruction


def spectral_smooth(u, L, cutoff_fraction=0.5):
    """
    Optional: apply spectral smoothing to remove high-frequency noise.

    Parameters:
    -----------
    u: ndarray (M, N) - State Space to smooth
    L: float - Length of domain
    cutoff_fraction: float - Fraction of modes to keep (0.5 = keep half)

    Returns:
    --------
    u_smooth: ndarray - Smoothed field
    """
    M, N = u.shape
    u_hat = fft(u, axis=1)

    # Create low-pass filter
    k_max = int(N * cutoff_fraction / 2)
    filter_mask = np.zeros(N)
    filter_mask[:k_max] = 1
    filter_mask[-k_max:] = 1

    # Apply filter
    u_hat_filtered = u_hat * filter_mask
    u_smooth = np.real(ifft(u_hat_filtered, axis=1))

    return u_smooth


def gaussian_smooth(u, sigma=1.0):
    """
    Optional: apply Gaussian smoothing in physical space.

    Parameters:
    -----------
    u: ndarray (M, N) - State Space to smooth
    sigma: float - Standard deviation of Gaussian kernel

    Returns:
    --------
    u_smooth: ndarray - Smoothed field
    """
    from scipy.ndimage import gaussian_filter1d

    u_smooth = gaussian_filter1d(u, sigma=sigma, axis=1, mode="wrap")
    return u_smooth
