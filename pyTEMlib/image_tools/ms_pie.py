import os
import numpy as np
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from scipy.ndimage import rotate, zoom


def mask_ring_cutoff(N, inner_radius=0, outer_radius=None):
    """Generate a binary circular mask centered in an N x N array."""
    if outer_radius is None:
        outer_radius = N / 2
    coords = np.arange(N)
    y, x = np.meshgrid(coords, coords, indexing="ij")
    center = (N - 1) / 2
    r = np.sqrt((x - center) ** 2 + (y - center) ** 2)
    return (r >= inner_radius) & (r <= outer_radius)


def upsample_nearest(array, factor):
    """Nearest-neighbor upsample by an integer factor."""
    return np.repeat(np.repeat(array, factor, axis=0), factor, axis=1)


def downsample_box(array, factor):
    """Box-filter downsample by an integer factor."""
    shape = array.shape
    return array.reshape(shape[0] // factor, factor, shape[1] // factor, factor).mean(axis=(1, 3))


def shift_wavefunction(wavefunction, dx_phys, dy_phys, pixel_size):
    """Fourier-domain shift equivalent to the original MATLAB `shift`.

    dx_phys, dy_phys are shifts in physical units (same units as pixel_size).
    """
    Ny, Nx = wavefunction.shape
    dk_y = 1.0 / (pixel_size * Ny)
    dk_x = 1.0 / (pixel_size * Nx)

    ky = np.linspace(-np.floor(Ny / 2), np.ceil(Ny / 2) - 1, Ny)
    kx = np.linspace(-np.floor(Nx / 2), np.ceil(Nx / 2) - 1, Nx)
    kX, kY = np.meshgrid(kx, ky, indexing="xy")
    kX = kX * dk_x
    kY = kY * dk_y

    f = fftshift(fft2(ifftshift(wavefunction)))
    ph = np.exp(-2j * np.pi * (dx_phys * kX + dy_phys * kY))
    f = f * ph
    output = fftshift(ifft2(ifftshift(f)))
    return output


def post_process(obj, px, py, dx, margin_pixels=10):
    """Post-process similar to the MATLAB `postProcess` helper.

    - Upsamples object by 2, rotates, crops to scan extents, removes phase ramp,
      then downsamples by 2.
    """
    # rotate coordinates (MATLAB used sind/cosd with degrees)
    # Here rot_angle is assumed zero in many calls; keep API simple and
    # operate without rotation unless px/py already rotated by caller.
    px = np.asarray(px)
    py = np.asarray(py)

    # Upsample by 2 to reduce rotation artifacts (MATLAB: imresize(obj,2))
    obj_rot = zoom(obj, 2.0, order=1)

    # No explicit rotation angle argument here; if needed pass rotated px/py
    cen_rot = int(np.floor(obj_rot.shape[0] / 2))

    # compute integer grid positions for scan (dx halved because of upsample)
    dx_up = dx / 2.0
    px_i = np.round(px / dx_up).astype(int)
    py_i = np.round(py / dx_up).astype(int)

    y_lb = int(np.ceil(np.min(py_i) + cen_rot))
    y_ub = int(np.floor(np.max(py_i) + cen_rot))
    x_lb = int(np.ceil(np.min(px_i) + cen_rot))
    x_ub = int(np.floor(np.max(px_i) + cen_rot))

    # Clip to bounds
    y_lb = max(0, y_lb)
    x_lb = max(0, x_lb)
    y_ub = min(obj_rot.shape[0] - 1, y_ub)
    x_ub = min(obj_rot.shape[1] - 1, x_ub)

    obj_crop = obj_rot[y_lb : y_ub + 1, x_lb : x_ub + 1]

    # remove phase ramp
    obj_crop_phase = remove_phase_ramp(obj_crop, dx_up)
    obj_crop = np.abs(obj_crop) * np.exp(1j * obj_crop_phase)

    # downsample back to original size
    obj_crop = zoom(obj_crop, 0.5, order=1)
    return obj_crop


def apply_ring_cutoff(dp, inner, outer):
    N_dp = dp.shape[0]
    x = np.arange(-int(np.fix(N_dp / 2)), N_dp - int(np.fix(N_dp / 2)))
    Y, X = np.meshgrid(x, x, indexing="ij")
    R = np.sqrt(X ** 2 + Y ** 2)
    mask = np.ones((N_dp, N_dp), dtype=dp.dtype)
    mask[(R >= outer) | (R < inner)] = 0
    # apply to each frame
    for i in range(dp.shape[2]):
        for j in range(dp.shape[3]):
            temp = dp[:, :, i, j]
            temp[mask == 0] = 0
            dp[:, :, i, j] = temp
    return dp


def apply_circular_cutoff(dp, cutoff):
    N_dp = dp.shape[0]
    x = np.arange(-int(np.fix(N_dp / 2)), N_dp - int(np.fix(N_dp / 2)))
    Y, X = np.meshgrid(x, x, indexing="ij")
    R = np.sqrt(X ** 2 + Y ** 2)
    mask = (R < cutoff)
    for i in range(dp.shape[2]):
        for j in range(dp.shape[3]):
            temp = dp[:, :, i, j]
            temp[~mask] = 0
            dp[:, :, i, j] = temp
    return dp


def mask_ring_cutoff_antialiasing(N_dp, inner, outer):
    N = N_dp * 16
    x = np.arange(-int(np.fix(N / 2)), N - int(np.fix(N / 2)))
    Y, X = np.meshgrid(x, x, indexing="ij")
    R = np.sqrt((X / 16.0) ** 2 + (Y / 16.0) ** 2)
    temp = np.ones((N, N), dtype=float)
    temp[(R >= outer) | (R < inner)] = 0
    # downsample by resizing using zoom
    mask = zoom(temp, N_dp / N, order=1)
    return mask


def generate_probe(dx, N, voltage, alpha_max):
    """Translate of MATLAB generateProbe.m

    Returns a complex probe array of shape (N, N).
    """
    df = 0.0
    C3 = 0.0
    C5 = 0.0
    C7 = 0.0
    lambda_ang = 12.398 / np.sqrt((2 * 511.0 + voltage) * voltage)

    amax = alpha_max * 1e-3
    amin = 0.0

    klimitmax = amax / lambda_ang
    klimitmin = amin / lambda_ang
    dk = 1.0 / (dx * N)

    kx = np.linspace(-np.floor(N / 2), np.ceil(N / 2) - 1, N)
    kX, kY = np.meshgrid(kx, kx, indexing="xy")
    kX = kX * dk
    kY = kY * dk
    kR = np.sqrt(kX ** 2 + kY ** 2)

    mask = ((kR <= klimitmax) & (kR >= klimitmin)).astype(float)
    chi = -np.pi * lambda_ang * kR ** 2 * df + (
        np.pi / 2 * C3 * lambda_ang ** 3 * kR ** 4
        + np.pi / 3 * C5 * lambda_ang ** 5 * kR ** 6
        + np.pi / 4 * C7 * lambda_ang ** 7 * kR ** 8
    )
    phase = np.exp(-1j * chi)
    probe = mask * phase
    probe = fftshift(ifft2(ifftshift(probe)))
    probe = probe / np.sum(np.abs(probe))
    return probe


def remove_phase_ramp(input_arr, dx):
    """Estimate and remove a linear phase ramp from a complex image.

    Returns the residual phase image (phase - fitted_plane).
    """
    Ny, Nx = input_arr.shape
    y = np.linspace(-np.floor(Ny / 2), np.ceil(Ny / 2) - 1, Ny)
    x = np.linspace(-np.floor(Nx / 2), np.ceil(Nx / 2) - 1, Nx)
    X, Y = np.meshgrid(x, y, indexing="xy")
    X = X * dx
    Y = Y * dx
    phase_image = np.angle(input_arr)

    A = np.column_stack((X.ravel(), Y.ravel()))
    b = phase_image.ravel()
    # solve for [p10, p01] in least-squares sense
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    background = (X * sol[0]) + (Y * sol[1])
    output = phase_image - background
    return output


def calculate_scan_positions(N_scan_x, N_scan_y, scanStepSize_x, scanStepSize_y, rot_ang):
    ppx = np.linspace(-np.floor(N_scan_x / 2), np.ceil(N_scan_x / 2) - 1, N_scan_x) * scanStepSize_x
    ppy = np.linspace(-np.floor(N_scan_y / 2), np.ceil(N_scan_y / 2) - 1, N_scan_y) * scanStepSize_y
    ppX, ppY = np.meshgrid(ppx, ppy, indexing="xy")
    # rotation (degrees)
    rad = np.deg2rad(rot_ang)
    ppY_rot = ppX * (-np.sin(rad)) + ppY * np.cos(rad)
    ppX_rot = ppX * np.cos(rad) + ppY * np.sin(rad)
    return ppX_rot, ppY_rot


def run_msPIE(dp, px, py, dx, N, probe0, Niter, bin, fn=None, save_results=False, results_dir="msPIE_Results"):
    """Run the multi-segment Ptychographic Iterative Engine (msPIE).

    Parameters
    ----------
    dp : ndarray
        Diffraction pattern data. Accepted shapes: (4, N_scan) or (2, 2, N_scan).
    px : ndarray
        X scan positions in physical units.
    py : ndarray
        Y scan positions in physical units.
    dx : float
        Scan grid spacing in the same physical units as px/py.
    N : int
        Patch size for the probe/object ROI.
    probe0 : ndarray
        Initial probe estimate, shape (N, N).
    Niter : int
        Number of iterations.
    bin : float
        Outer radius for the circular mask in pixels.
    fn : int or str, optional
        Frame identifier used when saving results.
    save_results : bool
        If True, write results to disk.
    results_dir : str
        Directory for saving output files.

    Returns
    -------
    obj : ndarray
        Reconstructed complex object.
    probe : ndarray
        Reconstructed complex probe.
    """
    dp = np.asarray(dp)
    if dp.ndim == 3 and dp.shape[0:2] == (2, 2):
        dp_flat = dp.reshape(4, dp.shape[2])
    elif dp.ndim == 2 and dp.shape[0] == 4:
        dp_flat = dp
    else:
        raise ValueError("dp must have shape (4, N_scan) or (2, 2, N_scan)")

    N_scan = dp_flat.shape[1]
    dp_avg = np.mean(dp_flat, axis=1)

    dp2 = np.zeros((2, 2, N_scan), dtype=dp_flat.dtype)
    dp2[0, 0, :] = dp_flat[2, :]
    dp2[0, 1, :] = dp_flat[3, :]
    dp2[1, 0, :] = dp_flat[1, :]
    dp2[1, 1, :] = dp_flat[0, :]

    dp2 = np.sqrt(dp2)

    probe = probe0.astype(complex)
    probe *= np.sqrt(np.sum(dp_avg) / np.sum(np.abs(probe0) ** 2) / probe0.size)

    px = np.asarray(px).ravel()
    py = np.asarray(py).ravel()
    if px.shape[0] != N_scan or py.shape[0] != N_scan:
        raise ValueError("px and py must have length equal to the number of scan points")

    half_floor = N // 2
    half_ceil = N - half_floor

    py_i = np.round(py / dx).astype(int)
    py_f = py - py_i * dx
    px_i = np.round(px / dx).astype(int)
    px_f = px - px_i * dx

    ind_obj_center = N * 2 // 2
    Ny_max = max(abs(np.round(np.min(py) / dx) - np.floor(N / 2)), abs(np.round(np.max(py) / dx) + np.ceil(N / 2))) * 2 + 1
    Nx_max = max(abs(np.round(np.min(px) / dx) - np.floor(N / 2)), abs(np.round(np.max(px) / dx) + np.ceil(N / 2))) * 2 + 1
    N_obj = int(max(Ny_max, Nx_max) + 1)

    obj = np.ones((N_obj, N_obj), dtype=complex)
    obj /= np.abs(obj)

    ind_x_lb = px_i - half_floor + ind_obj_center
    ind_x_ub = px_i + half_ceil - 1 + ind_obj_center
    ind_y_lb = py_i - half_floor + ind_obj_center
    ind_y_ub = py_i + half_ceil - 1 + ind_obj_center

    alpha = 0.1
    beta = 1.0
    eps = np.finfo(np.float64).eps

    mask = mask_ring_cutoff(N, 0, bin).astype(float)
    m2 = upsample_nearest(mask, 2)

    if save_results and fn is None:
        fn = 0

    if save_results:
        os.makedirs(results_dir, exist_ok=True)

    for iteration in range(1, Niter + 1):
        update_order = np.random.permutation(N_scan)
        for ind in update_order:
            probe = shift_wavefunction(probe, px_f[ind], py_f[ind], dx)

            obj_roi = obj[ind_y_lb[ind] : ind_y_ub[ind] + 1, ind_x_lb[ind] : ind_x_ub[ind] + 1]
            # Ensure obj_roi and probe have the same shape (pad/crop if needed)
            if obj_roi.shape != probe.shape:
                # center obj_roi into a buffer matching probe size
                buf = np.ones_like(probe, dtype=complex)
                h_obj, w_obj = obj_roi.shape
                h_p, w_p = probe.shape
                sy = max(0, (h_p - h_obj) // 2)
                sx = max(0, (w_p - w_obj) // 2)
                # limit copy to buffer bounds
                copy_h = min(h_obj, h_p - sy)
                copy_w = min(w_obj, w_p - sx)
                buf[sy:sy+copy_h, sx:sx+copy_w] = obj_roi[0:copy_h, 0:copy_w]
                obj_roi = buf
            psi = obj_roi * probe
            psi_old = psi.copy()
            psi = fftshift(fft2(psi))

            psi_up = upsample_nearest(psi * mask, 2)
            psi_cut = psi_up[2:, 2:]
            B = np.sqrt(
                np.sum(
                    np.abs(psi_cut).reshape(2, N - 1, 2, N - 1) ** 2,
                    axis=(1, 3),
                )
            )
            C = dp2[:, :, ind]
            k1 = C / (B + eps)

            k_temp = upsample_nearest(k1, N - 1)
            k_temp3 = np.zeros((2 * N, 2 * N), dtype=complex)
            k_temp3[2:, 2:] = k_temp
            kup = k_temp3 * m2 * 2.0
            k2 = downsample_box(kup, 2)

            psi = psi * k2
            psi = ifft2(ifftshift(psi))

            obj_roi_old = obj_roi.copy()
            probe_norm = np.max(np.abs(probe) ** 2)
            obj_roi += alpha * np.conj(probe) / (probe_norm + eps) * (psi - psi_old)
            # write back only the overlapping region from obj_roi into obj
            # destination bounds in object
            dest_y0 = int(ind_y_lb[ind])
            dest_x0 = int(ind_x_lb[ind])
            dest_y1 = int(min(ind_y_ub[ind] + 1, obj.shape[0]))
            dest_x1 = int(min(ind_x_ub[ind] + 1, obj.shape[1]))
            dest_h = dest_y1 - dest_y0
            dest_w = dest_x1 - dest_x0

            # source block size to copy (can't exceed dest or src sizes)
            src_h = min(dest_h, obj_roi.shape[0])
            src_w = min(dest_w, obj_roi.shape[1])
            src_sy = max(0, (obj_roi.shape[0] - src_h) // 2)
            src_sx = max(0, (obj_roi.shape[1] - src_w) // 2)

            src_block = obj_roi[src_sy:src_sy + src_h, src_sx:src_sx + src_w]
            obj[dest_y0:dest_y0 + src_h, dest_x0:dest_x0 + src_w] = src_block

            if iteration > 1:
                obj_roi_norm = np.max(np.abs(obj_roi_old) ** 2)
                probe += beta * np.conj(obj_roi_old) / (obj_roi_norm + eps) * (psi - psi_old)

            probe = shift_wavefunction(probe, -px_f[ind], -py_f[ind], dx)

        if save_results and (iteration == 1 or iteration % 10 == 0):
            obj_crop = post_process(obj, px, py, dx)
            filepath = os.path.join(results_dir, f"msPIE_{fn}frame_Iter{iteration}.npz")
            np.savez(filepath, obj=obj, obj_crop=obj_crop, probe=probe, dx=dx, Niter=Niter, bin=bin, alpha=alpha, beta=beta)

    return obj, probe


if __name__ == "__main__":
    # Example driver stub: user should replace synthetic values with real data.
    N = 64
    Niter = 20
    dx = 1.0
    bin = 16
    px = np.linspace(-5, 5, 8)
    py = np.linspace(-5, 5, 8)
    px, py = np.broadcast_arrays(px, py)
    px = px.ravel()
    py = py.ravel()

    dp = np.random.rand(4, px.size)
    probe0 = np.ones((N, N), dtype=complex)

    obj, probe = run_msPIE(dp, px, py, dx, N, probe0, Niter, bin, fn=0, save_results=False)
    print("msPIE completed. Object shape:", obj.shape, "Probe shape:", probe.shape)
