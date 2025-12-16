### consider conditional diffusion models for generalizability to multi-front scenarios

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from scipy.fft import fft, ifft
from sklearn.preprocessing import MinMaxScaler
import copy
import warnings

from DASHREDutils import *    

warnings.filterwarnings("ignore")

# Set device
device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
print(f"Using device: {device}")

# Set random seeds
np.random.seed(42)
torch.manual_seed(42)



if __name__ == "__main__":
    # HYPERPARAMETERS - Easy to tune

    # PDE parameters
    L = 32 * np.pi
    N = 256
    nu = 1.0
    mu_damping = 0.05  # Missing physics term
    dt = 0.02
    T = 100.0
    save_every = 3

    # SHRED/DA-SHRED parameters
    num_sensors = 15
    lags = 25
    hidden_size = 10
    decoder_layers = [128, 128]

    # Training parameters
    shred_epochs = 250
    shred_patience = 100
    dashred_epochs = 150
    dashred_patience = 100
    gan_epochs = 100
    smoothness_weight = 0.05  # Weight for smoothness regularization (try 0.0 to 0.2)

    # SINDy refinement parameters
    sindy_initial_threshold = 0.01
    sindy_max_iter = 50
    sindy_refinement_steps = 8
    sindy_threshold_increment = (
        0.04  # How much to increase in threshold on each refinement step
    )

    # Post-processing smoothing (optional)
    apply_smoothing = False  # Set to True to apply spectral smoothing
    smoothing_cutoff = (
        0.7  # Fraction of Fourier modes to keep (higher = less smoothing)
    )

    # Main execution

    # Generate initial condition
    x = np.linspace(0, L, N, endpoint=False)

    init_seeds = [16, 17, 18, 19, 20]
    u0_s = []
    for init_seed in init_seeds:
        u0 = np.cos(x / init_seed) * (1 + np.sin(x / init_seed))
        u0_s.append(u0)

    # Generate data
    print("\n[1] Generating PDE data...")
    print(f"    Simulation: Undamped KS (μ=0)")
    print(f"    Real Physics: Damped KS (μ={mu_damping})")

    ks_sim = KuramotoSivashinsky1D(L=L, N=N, nu=nu, mu=0.0, dt=dt)
    ks_real = KuramotoSivashinsky1D(L=L, N=N, nu=nu, mu=mu_damping, dt=dt)
    U_sim_list = []
    U_real_list = []

    for u0 in u0_s:
        U_sim_single = ks_sim.simulate(u0, T, save_every)  # shape: (nt, N)
        U_real_single = ks_real.simulate(u0, T, save_every)  # shape: (nt, N)

        U_sim_list.append(U_sim_single)
        U_real_list.append(U_real_single)

    U_sim = np.vstack(U_sim_list)
    U_real = np.vstack(U_real_list)
    # Plot KS heatmap
    t_sim = np.arange(0, T + dt, dt * save_every)
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(
        U_sim.T,
        aspect="auto",
        cmap="RdBu_r",
        extent=[t_sim[0], t_sim[-1], 0, L],
        origin="lower",
        interpolation="bilinear",
    )
    ax.set_xlabel("Time (t)")
    ax.set_ylabel("Space (x)")
    ax.set_title("1D Kuramoto-Sivashinsky Heatmap (Undamped Simulation)")
    plt.colorbar(im, ax=ax, label="u value")
    plt.tight_layout()
    plt.savefig("ks_heatmap.png", dpi=150, bbox_inches="tight")
    plt.show()

    print(f"    Data shape: {U_sim.shape} (timesteps x spatial points)")

    # Create datasets
    print("\n[2] Preparing datasets...")
    sensor_indices = np.linspace(0, N - 1, num_sensors, dtype=int)
    print(f"    Sensors: {num_sensors} at indices {sensor_indices}")
    print(f"    Time lags: {lags}")

    n_train = int(0.8 * len(U_sim))

    train_sim = TimeSeriesDataset(
        U_sim[:n_train], sensor_indices, lags, fit_scaler=True
    )
    valid_sim = TimeSeriesDataset(
        U_sim[n_train:], sensor_indices, lags, scaler=train_sim.get_scalers()
    )
    train_real = TimeSeriesDataset(
        U_real[:n_train], sensor_indices, lags, scaler=train_sim.get_scalers()
    )
    valid_real = TimeSeriesDataset(
        U_real[n_train:], sensor_indices, lags, scaler=train_sim.get_scalers()
    )

    print(f"    Training samples: {len(train_sim)}")
    print(f"    Validation samples: {len(valid_sim)}")

    # Initialize and train SHRED
    print("\n[3] Training SHRED on simulation data...")
    shred_model = SHRED(
        num_sensors=num_sensors,
        lags=lags,
        hidden_size=hidden_size,
        output_size=N,
        num_lstm_layers=3,
        decoder_layers=decoder_layers,
        dropout=0.1,
    )
    print(f"    Model parameters: {sum(p.numel() for p in shred_model.parameters()):,}")

    train_shred(
        shred_model, train_sim, valid_sim, epochs=shred_epochs, patience=shred_patience
    )

    # Evaluate SIM2REAL gap
    print("\n[4] Evaluating SIM2REAL gap...")
    scaler_U, _ = train_sim.get_scalers()

    _, _, mse_sim = evaluate(shred_model, valid_sim, scaler_U)
    _, _, mse_before = evaluate(shred_model, valid_real, scaler_U)
    print(f"    SHRED on simulation: RMSE = {np.sqrt(mse_sim):.6f}")
    print(f"    SHRED on real (gap): RMSE = {np.sqrt(mse_before):.6f}")
    print(f"    Gap ratio: {mse_before / mse_sim:.2f}x")

    # Initiate and train DASHRED
    dashred_model = DASHRED(shred_model, freeze_decoder=False)

    print("\n[5] Training DA-SHRED to close SIM2REAL gap...")
    train_hist_da, valid_hist_da = train_dashred(
        dashred_model,
        train_real,
        valid_real,
        sensor_indices,
        shred_model=shred_model,
        train_sim=train_sim,
        train_real=train_real,
        epochs=dashred_epochs,
        patience=dashred_patience,
        gan_epochs=gan_epochs,
        smoothness_weight=smoothness_weight,
    )

    # Evaluate DA-SHRED
    print("\n[6] Evaluating DA-SHRED...")
    pred_before, truth_real, mse_before = evaluate(shred_model, valid_real, scaler_U)
    pred_after, _, mse_after = evaluate(
        dashred_model, valid_real, scaler_U, is_dashred=True
    )

    # Optional: apply post-processing smoothing
    if apply_smoothing:
        print(f"    Applying spectral smoothing (cutoff={smoothing_cutoff})...")
        pred_after = spectral_smooth(pred_after, L, cutoff_fraction=smoothing_cutoff)
        mse_after = np.mean((pred_after - truth_real) ** 2)

    print(f"    SHRED on simulation: RMSE = {np.sqrt(mse_sim):.6f}")
    print(f"    SHRED on real (gap): RMSE = {np.sqrt(mse_before):.6f}")
    print(f"    DASHRED on real: RMSE = {np.sqrt(mse_after):.6f}")
    print(f"    Gap reduction: {(mse_before - mse_after) / mse_before * 100:.1f}%")
    print(f"    Improvement: {mse_before / mse_after:.2f}x")

    # Reconstruction comparison

    print("\n[6b] Generating three-panel reconstruction comparison...")

    dashred_model.eval()
    train_loader_real = DataLoader(train_real, batch_size=len(train_real))
    with torch.no_grad():
        for sensors, _ in train_loader_real:
            sensors = sensors.to(device)
            pred_train_dashred, _, _ = dashred_model(sensors, apply_transform=True)
            pred_train_dashred = scaler_U.inverse_transform(
                pred_train_dashred.cpu().numpy()
            )

    # Optional smoothing for visualization
    if apply_smoothing:
        pred_train_dashred = spectral_smooth(
            pred_train_dashred, L, cutoff_fraction=smoothing_cutoff
        )

    U_sim_train = U_sim[lags:n_train]
    U_real_train = U_real[lags:n_train]
    t_train = np.arange(lags, n_train) * dt * save_every

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    vmin = min(U_sim_train.min(), U_real_train.min(), pred_train_dashred.min())
    vmax = max(U_sim_train.max(), U_real_train.max(), pred_train_dashred.max())

    im1 = axes[0].imshow(
        U_sim_train.T,
        aspect="auto",
        cmap="RdBu_r",
        extent=[t_train[0], t_train[-1], 0, L],
        origin="lower",
        vmin=vmin,
        vmax=vmax,
        interpolation="bilinear",
    )
    axes[0].set_xlabel("Time (t)", fontsize=12)
    axes[0].set_ylabel("Space (x)", fontsize=12)
    axes[0].set_title("Simulation (Undamped KS)", fontsize=14)
    plt.colorbar(im1, ax=axes[0], label="u")

    im2 = axes[1].imshow(
        U_real_train.T,
        aspect="auto",
        cmap="RdBu_r",
        extent=[t_train[0], t_train[-1], 0, L],
        origin="lower",
        vmin=vmin,
        vmax=vmax,
        interpolation="bilinear",
    )
    axes[1].set_xlabel("Time (t)", fontsize=12)
    axes[1].set_ylabel("Space (x)", fontsize=12)
    axes[1].set_title("Real Physics (Damped KS, μ={})".format(mu_damping), fontsize=14)
    plt.colorbar(im2, ax=axes[1], label="u")

    im3 = axes[2].imshow(
        pred_train_dashred.T,
        aspect="auto",
        cmap="RdBu_r",
        extent=[t_train[0], t_train[-1], 0, L],
        origin="lower",
        vmin=vmin,
        vmax=vmax,
        interpolation="bilinear",
    )
    axes[2].set_xlabel("Time (t)", fontsize=12)
    axes[2].set_ylabel("Space (x)", fontsize=12)
    axes[2].set_title("DA-SHRED Reconstruction", fontsize=14)
    plt.colorbar(im3, ax=axes[2], label="u")

    plt.tight_layout()
    plt.savefig("reconstruction_comparison.png", dpi=150, bbox_inches="tight")
    print("    Saved: reconstruction_comparison.png")
    plt.show()

    # SINDy: Discovery of Missing Physics

    print("\n[7] Identifying missing physics with SINDy...")

    Xi, feature_names = identify_missing_damping(
        shred_model,
        dashred_model,
        train_sim,
        train_real,
        x,
        L,
        dt,
        save_every,
        initial_threshold=sindy_initial_threshold,
        max_iter=sindy_max_iter,
        refinement_steps=sindy_refinement_steps,
        threshold_increment=sindy_threshold_increment,
    )

    # Visualize SINDy coefficients
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(Xi, cmap="RdBu_r", aspect="auto")
    ax.set_xticks(range(Xi.shape[1]))
    ax.set_xticklabels([f"dz{i}" for i in range(Xi.shape[1])])
    ax.set_yticks(range(len(feature_names)))
    ax.set_yticklabels(feature_names)
    ax.set_title("SINDy Coefficients for Latent Discrepancy")
    plt.colorbar(im, ax=ax, label="Coefficient")
    plt.tight_layout()
    plt.savefig("sindy_coefficients.png", dpi=150)
    plt.show()

    # Plot final results

    print("\n[8] Generating plots...")

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle("Prediction results on unseen sensor data", fontsize=14, y=0.98)

    scaler_U, _ = train_sim.get_scalers()
    pred_before, _, mse_before = evaluate(shred_model, valid_real, scaler_U)
    pred_after, truth_real, mse_after = evaluate(
        dashred_model, valid_real, scaler_U, is_dashred=True
    )

    if apply_smoothing:
        pred_after = spectral_smooth(pred_after, L, cutoff_fraction=smoothing_cutoff)

    for i, idx in enumerate([0, len(truth_real) // 2, len(truth_real) - 1]):
        axes[0, i].plot(x, truth_real[idx], "b-", lw=2, label="Ground Truth")
        axes[0, i].plot(x, pred_before[idx], "r--", lw=1.5, label="SHRED (before DA)")
        axes[0, i].plot(x, pred_after[idx], "g--", lw=1.5, label="DA-SHRED")
        axes[0, i].set_xlabel("x")
        axes[0, i].set_ylabel("u")
        axes[0, i].set_title(f"Sample {idx}")
        axes[0, i].legend(fontsize=8)
        axes[0, i].grid(alpha=0.3)

    err_before = np.sqrt(np.mean((pred_before - truth_real) ** 2, axis=1))
    err_after = np.sqrt(np.mean((pred_after - truth_real) ** 2, axis=1))

    axes[1, 0].plot(err_before, "r-", label="SHRED (before DA)")
    axes[1, 0].plot(err_after, "g-", label="DA-SHRED")
    axes[1, 0].set_xlabel("Time step")
    axes[1, 0].set_ylabel("RMSE")
    axes[1, 0].set_title("Reconstruction Error")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    im = axes[1, 1].imshow(Xi, cmap="RdBu_r", aspect="auto")
    axes[1, 1].set_xlabel("Latent Dim")
    axes[1, 1].set_ylabel("Library Feature")
    axes[1, 1].set_title("SINDy Coefficients (Missing Physics)")
    plt.colorbar(im, ax=axes[1, 1])

    axes[1, 2].imshow(
        np.abs(pred_after - truth_real).T,
        aspect="auto",
        cmap="hot",
        extent=[0, len(truth_real), 0, L],
    )
    axes[1, 2].set_xlabel("Time")
    axes[1, 2].set_ylabel("x")
    axes[1, 2].set_title("DA-SHRED Absolute Error")

    axes[0, 1].annotate(
        "Snapshots (u(x))",
        xy=(0.5, 1.25),
        xycoords="axes fraction",
        ha="center",
        va="center",
        fontsize=12,
    )
    axes[1, 1].annotate(
        "Error / Coef / Heatmap",
        xy=(0.5, 1.25),
        xycoords="axes fraction",
        ha="center",
        va="center",
        fontsize=12,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    plt.savefig("dashred_results.png", dpi=150, bbox_inches="tight")
    print("    Saved: dashred_results.png")
    plt.show()

    # Save checkpoint
    torch.save(
        {
            "shred": shred_model.state_dict(),
            "dashred": dashred_model.state_dict(),
            "sensor_indices": sensor_indices,
            "params": {
                "num_sensors": num_sensors,
                "lags": lags,
                "hidden_size": hidden_size,
            },
        },
        "dashred_checkpoint.pt",
    )
    print("    Saved: dashred_checkpoint.pt")

    print("\nDone!")
