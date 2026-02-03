"""
Utility functions for loading grid coordinates and plotting spatial fields
with proper xy positioning.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pathlib import Path
from typing import Optional, Tuple


def load_xy_grid(
    grid_file: str = "grid_xy.npy",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load xy grid coordinates from file.

    Parameters
    ----------
    grid_file : str
        Path to the grid_xy.npy file containing (2, n_points) array
        where first row is x coords and second row is y coords.

    Returns
    -------
    x : np.ndarray
        1D array of x coordinates for all spatial points
    y : np.ndarray
        1D array of y coordinates for all spatial points
    """
    grid_xy = np.load(grid_file)
    if grid_xy.shape[0] != 2:
        raise ValueError(
            f"Expected grid_xy shape (2, n_points), got {grid_xy.shape}"
        )
    x = grid_xy[0, :]
    y = grid_xy[1, :]
    return x, y


def load_grid_dims(
    dims_file: str = "coarse_grid_dims.txt",
) -> Tuple[int, int]:
    """
    Load coarse grid dimensions (nx, ny) from file.

    Parameters
    ----------
    dims_file : str
        Path to the coarse_grid_dims.txt file

    Returns
    -------
    nx : int
        Number of grid points in x direction
    ny : int
        Number of grid points in y direction
    """
    with open(dims_file, "r") as f:
        line = f.readline()
        nx, ny = map(int, line.split())
    return nx, ny


def get_xy_extents(
    x: np.ndarray, y: np.ndarray
) -> Tuple[float, float, float, float]:
    """
    Get extent tuple (xmin, xmax, ymin, ymax) from xy coordinates.

    Parameters
    ----------
    x : np.ndarray
        1D array of x coordinates
    y : np.ndarray
        1D array of y coordinates

    Returns
    -------
    extent : tuple
        (xmin, xmax, ymin, ymax)
    """
    xmin, xmax = np.min(x), np.max(x)
    ymin, ymax = np.min(y), np.max(y)
    return (xmin, xmax, ymin, ymax)



def straighten_data(data, nx: int, ny: int, shift_per_row):
    """
    Straighten x-direction rotation over time via row-wise circular *fractional* shifts.

    Args:
        data: array of shape (T, ny*nx) or (T, ny, nx); T = time steps
        nx, ny: grid dimensions
        shift_per_row: scalar (pixels per timestep for every row) or
                       array of shape (ny,) giving pixels-per-timestep per row

    Returns:
        straightened: array of shape (T, ny, nx)
    """
    # Normalize input shape -> (T, ny, nx)
    if data.ndim == 2 and data.shape[1] == nx * ny:
        frames = data.reshape(data.shape[0], ny, nx)
    elif data.ndim == 3 and data.shape[1:] == (ny, nx):
        frames = data
    else:
        raise ValueError("data must be (T, ny*nx) or (T, ny, nx)")

    T = frames.shape[0]

    # Normalize shift_per_row to shape (ny,)
    s = np.asarray(shift_per_row, dtype=float)
    if s.ndim == 0:
        s = np.full(ny, float(s))
    elif s.shape != (ny,):
        raise ValueError("shift_per_row must be scalar or shape (ny,)")

    # Precompute frequency vector for fractional circular shifts along x
    # Using FFT-based phase shift: x(t - shift) <-> X(k) * exp(-i*2π*k*shift/N)
    freqs = np.fft.fftfreq(nx)  # shape (nx,)

    out = np.empty((T, ny, nx), dtype=frames.dtype)

    for t in range(T):
        img = frames[t]  # (ny, nx)
        shifts = t * s   # pixels to shift each row at time t, shape (ny,)

        # Broadcast row-wise phase factors: (ny, nx)
        phase = np.exp(-2j * np.pi * freqs[None, :] * shifts[:, None])

        # FFT along x (axis=1), apply per-row phase, inverse FFT
        X = np.fft.fft(img, axis=1)
        shifted = np.fft.ifft(X * phase, axis=1).real

        # Cast back to original dtype if needed
        if np.issubdtype(out.dtype, np.integer):
            shifted = np.rint(shifted)

        out[t] = shifted.astype(out.dtype, copy=False)

    return out


def plot_snapshots_coarse(
    A,
    nx: int,
    ny: int,
    x_span,
    y_span,
    field_name: str,
    output_dir: Path,
):
    """
    Plots and saves snapshots of the data at different time indices.

    Handles both direct numpy arrays and file paths (for memmap compatibility).

    Parameters
    ----------
    A : numpy.ndarray or Path-like or str
        Data array of shape (n_spatial, n_timesteps) or path to .npy file.
        If path, will load with mmap_mode='r' for memory efficiency.
    nx : int
        Number of grid points in x direction
    ny : int
        Number of grid points in y direction
    x_span : tuple or list
        x-range [x_min, x_max]
    y_span : tuple or list
        y-range [y_min, y_max]
    field_name : str
        Name of the field for title and filename
    output_dir : Path or str
        Output directory for saving the figure

    Returns
    -------
    None
        Saves figure to disk and displays it
    """
    # Load data if path provided
    if isinstance(A, (str, Path)):
        A = np.load(A, mmap_mode='r')
        print(f"Loaded data from {A}: shape (n_spatial={A.shape[0]}, "
              f"n_temporal={A.shape[1]})")
        # Reshape from (n_spatial, n_temporal) to (n_temporal, ny, nx)
        A = A.T.reshape(-1, ny, nx)
    else:
        print(f"Data shape received by plot_snapshots_coarse: {A.shape}")

    # Handle both (n_timesteps, ny, nx) and (n_spatial, n_temporal) shapes
    if A.ndim == 3:
        n_timesteps = A.shape[0]
    else:
        raise ValueError(
            f"Expected 3D array, got shape {A.shape}"
        )

    idx = [0, n_timesteps // 2, n_timesteps - 1]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    fig.suptitle(f"Snapshots for field: {field_name}", fontsize=16)
    for i, (ax, t) in enumerate(zip(axes, idx)):
        values = np.asarray(A[t, :, :]).reshape(ny, nx)
        im = ax.imshow(
            values,
            origin="lower",
            extent=[x_span[0], x_span[1], y_span[0], y_span[1]],
            aspect="auto",
        )
        ax.set_title(["First", "Middle", "Last"][i])
        ax.set_xlabel("X")
        if i == 0:
            ax.set_ylabel("Y")
        fig.colorbar(im, ax=ax)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = Path(output_dir) / f"{field_name}_snapshots_loaded.png"
    plt.savefig(out)
    print(f"Saved snapshots to {out}")
    plt.close(fig)


def plot_spatial_mode(
    mode_vector: np.ndarray,
    nx: int,
    ny: int,
    x: Optional[np.ndarray] = None,
    y: Optional[np.ndarray] = None,
    title: str = "Spatial Mode",
    cmap: str = "viridis",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plot a spatial mode or field snapshot as a 2D image with xy coordinates.


    Parameters
    ----------
    mode_vector : np.ndarray
        1D array of shape (nx*ny,) containing the field values at each
        spatial point in flattened row-major order.
    nx : int
        Number of grid points in x direction
    ny : int
        Number of grid points in y direction
    x : np.ndarray, optional
        1D array of x coordinates for each spatial point. If provided,
        used to set extent for correct xy positioning.
    y : np.ndarray, optional
        1D array of y coordinates for each spatial point. If provided,
        used to set extent for correct xy positioning.
    title : str
        Title for the plot
    cmap : str
        Colormap name
    ax : plt.Axes, optional
        Matplotlib axes to plot on. If None, uses current axes.

    Returns
    -------
    ax : plt.Axes
        The axes object with the plotted image
    """
    if ax is None:
        ax = plt.gca()

    # Reshape to 2D grid
    field_2d = mode_vector.reshape(ny, nx)

    # Prepare extent if coordinates provided
    extent = None
    if x is not None and y is not None:
        extent = (np.min(x), np.max(x), np.min(y), np.max(y))

    # Plot with origin='lower' to match standard xy coordinate system
    im = ax.imshow(
        field_2d,
        aspect="auto",
        origin="lower",
        cmap=cmap,
        extent=extent,
    )
    ax.set_title(title)

    # Set axis labels
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    # Add colorbar
    plt.colorbar(im, ax=ax)

    return ax


def plot_field_snapshot(
    P_mat: np.ndarray,
    time_index: int,
    nx: int,
    ny: int,
    x: Optional[np.ndarray] = None,
    y: Optional[np.ndarray] = None,
    title: str = None,
    cmap: str = "jet",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plot a snapshot of a field at a given time index with xy coordinates.

    Parameters
    ----------
    P_mat : np.ndarray
        Pressure/field matrix of shape (n_spatial, n_temporal) where each
        column is a time snapshot.
    time_index : int
        Index of the time snapshot to plot (0 to n_temporal-1)
    nx : int
        Number of grid points in x direction
    ny : int
        Number of grid points in y direction
    x : np.ndarray, optional
        1D array of x coordinates for each spatial point
    y : np.ndarray, optional
        1D array of y coordinates for each spatial point
    title : str, optional
        Title for the plot. If None, defaults to "Field at t={time_index}"
    cmap : str
        Colormap name
    ax : plt.Axes, optional
        Matplotlib axes to plot on. If None, uses current axes.

    Returns
    -------
    ax : plt.Axes
        The axes object with the plotted image
    """
    if ax is None:
        ax = plt.gca()

    if title is None:
        title = f"Field at t={time_index}"

    # Extract snapshot
    snapshot = np.asarray(P_mat[:, time_index])

    # Use plot_spatial_mode for consistency
    ax = plot_spatial_mode(
        snapshot,
        nx=nx,
        ny=ny,
        x=x,
        y=y,
        title=title,
        cmap=cmap,
        ax=ax,
    )

    return ax


def plot_multiple_snapshots(
    P_mat: np.ndarray,
    time_indices: list,
    nx: int,
    ny: int,
    x: Optional[np.ndarray] = None,
    y: Optional[np.ndarray] = None,
    field_name: str = "Field",
    cmap: str = "jet",
    figsize: Optional[Tuple[int, int]] = None,
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Plot multiple field snapshots in a grid layout.

    Parameters
    ----------
    P_mat : np.ndarray
        Pressure/field matrix of shape (n_spatial, n_temporal)
    time_indices : list
        List of time indices to plot
    nx : int
        Number of grid points in x direction
    ny : int
        Number of grid points in y direction
    x : np.ndarray, optional
        1D array of x coordinates for each spatial point
    y : np.ndarray, optional
        1D array of y coordinates for each spatial point
    field_name : str
        Name of the field for figure title
    cmap : str
        Colormap name
    figsize : tuple, optional
        Figure size. If None, defaults to (5*len(time_indices), 4)

    Returns
    -------
    fig : plt.Figure
        The figure object
    axs : np.ndarray
        Array of axes objects
    """
    if figsize is None:
        figsize = (5 * len(time_indices), 4)

    fig, axs = plt.subplots(
        1, len(time_indices), figsize=figsize, sharey=True
    )
    if len(time_indices) == 1:
        axs = np.array([axs])

    for ax, t_idx in zip(axs, time_indices):
        plot_field_snapshot(
            P_mat,
            time_index=t_idx,
            nx=nx,
            ny=ny,
            x=x,
            y=y,
            title=f"{field_name} at t={t_idx}",
            cmap=cmap,
            ax=ax,
        )

    fig.suptitle(f"Snapshots: {field_name}", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    return fig, axs


def plot_spatial_modes_with_xy(
    U: np.ndarray,
    S: np.ndarray,
    VT: np.ndarray,
    P_mat: np.ndarray,
    nx: int,
    ny: int,
    num_modes: int = 6,
    x: Optional[np.ndarray] = None,
    y: Optional[np.ndarray] = None,
    figsize: Optional[Tuple[int, int]] = None,
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Create comprehensive POD analysis plot with xy-positioned spatial modes.

    Parameters
    ----------
    U : np.ndarray
        Left singular vectors (spatial modes) from SVD, shape
        (n_spatial, n_modes)
    S : np.ndarray
        Singular values from SVD
    VT : np.ndarray
        Right singular vectors (temporal modes), shape (n_modes, n_temporal)
    P_mat : np.ndarray
        Pressure matrix of shape (n_spatial, n_temporal)
    nx : int
        Number of grid points in x direction
    ny : int
        Number of grid points in y direction
    num_modes : int
        Number of modes to display (default: 6)
    x : np.ndarray, optional
        1D array of x coordinates for each spatial point
    y : np.ndarray, optional
        1D array of y coordinates for each spatial point
    figsize : tuple, optional
        Figure size. If None, defaults to (3*num_modes, 12)

    Returns
    -------
    fig : plt.Figure
        The figure object
    axs : np.ndarray
        2D array of axes objects (3, num_modes)
    """
    if figsize is None:
        figsize = (3 * num_modes, 12)

    num_modes = min(num_modes, U.shape[1], VT.shape[0])
    fig, axs = plt.subplots(3, num_modes, figsize=figsize)

    if num_modes == 1:
        axs = axs.reshape(3, 1)

    for i in range(num_modes):
        # Row 0: Spatial modes with xy coordinates
        plot_spatial_mode(
            U[:, i],
            nx=nx,
            ny=ny,
            x=x,
            y=y,
            title=f"Spatial Mode {i+1}",
            cmap="viridis",
            ax=axs[0, i],
        )

        # Row 1: Temporal modes
        axs[1, i].plot(VT[i, :])
        axs[1, i].set_title(f"Temporal Mode {i+1}")
        axs[1, i].set_xlabel("Time Index")
        axs[1, i].set_ylabel("Coefficient")
        axs[1, i].grid()

    # Row 2, Col 0: Singular values
    axs[2, 0].plot(S[:min(100, len(S))], marker="o")
    axs[2, 0].set_title("Singular Values")
    axs[2, 0].set_xlabel("Mode Index")
    axs[2, 0].set_ylabel("Magnitude")
    axs[2, 0].grid()

    # Row 2, Col 1: Remaining variance
    rank_plot = np.arange(1, min(100, len(S)) + 1)
    remaining_variance = (
        1 - np.cumsum(S[: min(100, len(S))] ** 2) / np.sum(S**2)
    )
    axs[2, 1].semilogy(rank_plot[:-1], remaining_variance[:-1], "o-")
    axs[2, 1].set_title("Remaining Variance")
    axs[2, 1].set_xlabel("Number of Modes")
    axs[2, 1].set_ylabel("Remaining Variance (log scale)")
    axs[2, 1].grid()

    # Row 2, Col 2: First snapshot with xy coordinates
    plot_field_snapshot(
        P_mat,
        time_index=0,
        nx=nx,
        ny=ny,
        x=x,
        y=y,
        title="First Snapshot",
        cmap="jet",
        ax=axs[2, 2],
    )

    # Hide extra subplots if num_modes > 3
    for i in range(3, num_modes):
        axs[2, i].axis("off")

    plt.tight_layout()

    return fig, axs


def create_field_animation(
    P_mat: np.ndarray,
    nx: int,
    ny: int,
    output_file: str = "field_animation.mp4",
    x: Optional[np.ndarray] = None,
    y: Optional[np.ndarray] = None,
    field_name: str = "Field",
    cmap: str = "jet",
    fps: int = 15,
    figsize: Tuple[int, int] = (8, 6),
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> None:
    """
    Create and save an mp4 animation of a field over time.

    Parameters
    ----------
    P_mat : np.ndarray
        Field matrix of shape (n_spatial, n_temporal) where each
        column is a time snapshot.
    nx : int
        Number of grid points in x direction
    ny : int
        Number of grid points in y direction
    output_file : str, optional
        Path for output mp4 file (default: "field_animation.mp4")
    x : np.ndarray, optional
        1D array of x coordinates for each spatial point
    y : np.ndarray, optional
        1D array of y coordinates for each spatial point
    field_name : str, optional
        Name of the field for plot title (default: "Field")
    cmap : str, optional
        Colormap name (default: "jet")
    fps : int, optional
        Frames per second for animation (default: 15)
    figsize : tuple, optional
        Figure size (width, height) (default: (8, 6))
    vmin : float, optional
        Minimum value for colorbar. If None, uses data min.
    vmax : float, optional
        Maximum value for colorbar. If None, uses data max.

    Returns
    -------
    None
        Animation is saved to output_file
    """
    n_temporal = P_mat.shape[1]

    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    fig.suptitle(f"Animation: {field_name}", fontsize=14)

    # Prepare extent if coordinates provided
    extent = None
    if x is not None and y is not None:
        extent = (np.min(x), np.max(x), np.min(y), np.max(y))

    # Compute global vmin/vmax if not provided
    if vmin is None or vmax is None:
        global_min = np.nanmin(P_mat)
        global_max = np.nanmax(P_mat)
        if vmin is None:
            vmin = global_min
        if vmax is None:
            vmax = global_max

    # Create initial image
    first_frame = np.asarray(P_mat[:, 0]).reshape(ny, nx)
    im = ax.imshow(
        first_frame,
        aspect="auto",
        origin="lower",
        cmap=cmap,
        extent=extent,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(field_name)

    # Create time text
    time_text = ax.text(
        0.05,
        0.95,
        "",
        transform=ax.transAxes,
        va="top",
        ha="left",
        color="white",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="black", alpha=0.5),
    )

    def update(frame):
        """Update function for animation."""
        field_frame = np.asarray(P_mat[:, frame]).reshape(ny, nx)
        im.set_data(field_frame)

        # Update colorbar if values are very different from initial frame
        frame_min = np.nanmin(field_frame)
        frame_max = np.nanmax(field_frame)
        if frame_min < vmin or frame_max > vmax:
            # Keep static colorbar limits across frames
            pass

        time_text.set_text(f"Time Step: {frame}/{n_temporal-1}")
        return [im, time_text]

    # Create animation
    print(f"Creating animation with {n_temporal} frames at {fps} fps...")
    ani = FuncAnimation(
        fig,
        update,
        frames=n_temporal,
        blit=True,
        interval=1000 / fps,
        repeat=True,
    )

    # Save animation
    print(f"Saving animation to {output_file}...")
    try:
        ani.save(output_file, writer="ffmpeg", fps=fps, dpi=100)
        print(f"Animation saved successfully to {output_file}")
    except Exception as e:
        print(f"Error saving animation: {e}")
        print("Make sure ffmpeg is installed: brew install ffmpeg")

    plt.close(fig)


# ==================== SVD and Reconstruction Functions ====================

def compute_svd(data_mat):
    """
    Compute the Singular Value Decomposition of a data matrix.

    Parameters
    ----------
    data_mat : numpy.ndarray
        Input data matrix of shape (N_spatial, N_temporal)

    Returns
    -------
    U : numpy.ndarray
        Left singular vectors (spatial modes), shape (N_spatial, rank)
    S : numpy.ndarray
        Singular values, shape (rank,)
    VT : numpy.ndarray
        Right singular vectors (temporal modes), shape (rank, N_temporal)
    """
    U, S, VT = np.linalg.svd(data_mat, full_matrices=False)
    return U, S, VT


def reconstruct_pressure_matrix(U, S, VT, rank):
    """
    Reconstruct a data matrix from truncated SVD components.

    Parameters
    ----------
    U : numpy.ndarray
        Left singular vectors (spatial modes)
    S : numpy.ndarray
        Singular values
    VT : numpy.ndarray
        Right singular vectors (temporal modes)
    rank : int
        Number of modes to use in reconstruction

    Returns
    -------
    P_approx : numpy.ndarray
        Reconstructed matrix from rank-truncated SVD
    """
    Ur = U[:, :rank]
    Sr = np.diag(S[:rank])
    VTr = VT[:rank, :]
    P_approx = Ur @ Sr @ VTr
    return P_approx


def compare_reconstructions(data_mat, data_approx, rank, nx, ny):
    """
    Compare original and reconstructed data matrices with visualization.

    Parameters
    ----------
    data_mat : numpy.ndarray
        Original data matrix of shape (N_spatial, N_temporal)
    data_approx : numpy.ndarray
        Reconstructed data matrix (same shape as data_mat)
    rank : int
        Rank used for reconstruction (for title)
    nx : int
        Number of grid points in x direction
    ny : int
        Number of grid points in y direction

    Returns
    -------
    error : float
        Relative Frobenius norm error between original and reconstruction
    """
    error = (np.linalg.norm(data_mat - data_approx, ord="fro") /
             np.linalg.norm(data_mat, ord="fro"))

    N_spatial, N_temp = data_mat.shape

    # Select a few time indices to visualize
    idx = ([0, N_temp // 2, N_temp - 1] if N_temp >= 3
           else list(range(N_temp)))
    ncols = len(idx)

    fig, axs = plt.subplots(2, ncols, figsize=(4 * ncols, 8))
    if ncols == 1:
        axs = axs.reshape(2, 1)

    for j, t in enumerate(idx):
        orig_field = data_mat[:, t].reshape(ny, nx)
        recon_field = data_approx[:, t].reshape(ny, nx)

        # Use common color scaling for fair comparison
        vmin = min(orig_field.min(), recon_field.min())
        vmax = max(orig_field.max(), recon_field.max())

        im0 = axs[0, j].imshow(
            orig_field, aspect="auto", origin="lower",
            cmap="viridis", vmin=vmin, vmax=vmax
        )
        axs[0, j].set_title(f"Original t={t}")
        if j == 0:
            axs[0, j].set_ylabel("Y")

        im1 = axs[1, j].imshow(
            recon_field, aspect="auto", origin="lower",
            cmap="viridis", vmin=vmin, vmax=vmax
        )
        axs[1, j].set_title(f"Reconstructed t={t}")
        if j == 0:
            axs[1, j].set_ylabel("Y")

        fig.colorbar(im0, ax=axs[0, j])
        fig.colorbar(im1, ax=axs[1, j])

    plt.suptitle(
        f"Reconstruction with Rank {rank} (Relative Error: {error:.4f})",
        fontsize=14
    )
    plt.tight_layout()
    fig.savefig("reconstruction_comparison.png", dpi=150, bbox_inches="tight")
    print(f"Saved reconstruction comparison to reconstruction_comparison.png")
    print(f"Relative reconstruction error (rank {rank}): {error:.4f}")
    plt.show()

    return error

def save_movie_coarse(
    data_matrix_path: str,
    Nx: int,
    Ny: int,
    xspan: Tuple[float, float],
    yspan: Tuple[float, float],
    field_name: str,
    output_dir: str,
    fps: int = 15,
) -> None:
    """
    Create and save an mp4 animation from a data matrix file.

    Compatibility wrapper for shreddable_data_comlete.py save_movie_coarse.
    Loads data from file and creates animation with extent and labels.

    Parameters
    ----------
    data_matrix_path : str
        Path to data matrix .npy file of shape (n_spatial, n_temporal)
    Nx : int
        Number of grid points in x direction
    Ny : int
        Number of grid points in y direction
    xspan : tuple
        (xmin, xmax) for extent
    yspan : tuple
        (ymin, ymax) for extent
    field_name : str
        Name of the field for title and colorbar label
    output_dir : str
        Directory to save output mp4 file
    fps : int, optional
        Frames per second (default: 15)

    Returns
    -------
    None
        Animation saved as {output_dir}/{field_name}_animation.mp4
    """
    from pathlib import Path

    data_matrix_path = Path(data_matrix_path)
    output_dir = Path(output_dir)

    # Load data
    A = np.load(data_matrix_path, mmap_mode="r")
    n_points, n_timesteps = A.shape

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.suptitle(f"Animation for field: {field_name}", fontsize=16)

    # Create initial image with extent
    vals0 = np.asarray(A[:, 0]).reshape(Ny, Nx)
    im = ax.imshow(
        vals0,
        origin="lower",
        extent=[xspan[0], xspan[1], yspan[0], yspan[1]],
        aspect="auto",
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(field_name)

    # Create time text
    time_text = ax.text(
        0.05,
        0.95,
        "",
        transform=ax.transAxes,
        va="top",
        ha="left",
        color="white",
    )

    def update(frame):
        """Update function for animation."""
        values = np.asarray(A[:, frame]).reshape(Ny, Nx)
        im.set_data(values)
        im.set_clim(vmin=np.nanmin(values), vmax=np.nanmax(values))
        time_text.set_text(f"Timestep: {frame}/{n_timesteps-1}")
        return [im, time_text]

    # Create animation
    print(f"Creating animation with {n_timesteps} frames at {fps} fps...")
    ani = FuncAnimation(
        fig, update, frames=n_timesteps, blit=True, interval=1000 / fps
    )

    # Save animation
    output_file = output_dir / f"{field_name}_animation.mp4"
    print(f"Saving animation to {output_file}...")
    try:
        ani.save(output_file, writer="ffmpeg", fps=fps)
        print(f"Animation saved successfully to {output_file}")
    except Exception as e:
        print(f"Error saving animation: {e}")
        print("Make sure ffmpeg is installed: brew install ffmpeg")

    plt.close(fig)

def plot_histogram(data, title, xlabel, ylabel, bins=50):
    plt.figure(figsize=(8, 6))
    # Flatten the data if it's multidimensional
    data_flat = data.flatten() if hasattr(data, 'flatten') else np.array(data).flatten()
    plt.hist(data_flat, bins=bins, edgecolor='black', alpha=0.7)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(axis='y', alpha=0.75)
    plt.show()