import io
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import librosa
import librosa.display

st.set_page_config(page_title="Mnemos", layout="wide")
st.title("Mnemos: Music → Beats/Onsets → Spikes → EI Network")

MAX_SECONDS = 30.0
DEFAULT_DT_MS = 1.0

# ---------------------------
# Helpers
# ---------------------------

def run_brian_subprocess(drivetimes: np.ndarray, T: float, dt_ms: float = 1.0):
    """Run Brian2 simulation in a separate Python process to avoid Streamlit thread/signal issues."""
    payload = {
        "drivetimes": np.asarray(drivetimes, dtype=float).tolist(),
        "T": float(T),
        "dt_ms": float(dt_ms),
    }

    p = subprocess.run(
        [sys.executable, "brian_worker.py"],
        input=json.dumps(payload).encode("utf-8"),
        capture_output=True,
        check=False,
    )

    if p.returncode != 0:
        raise RuntimeError(
            "Brian worker failed.\n"
            f"STDOUT:\n{p.stdout.decode('utf-8', 'ignore')}\n\n"
            f"STDERR:\n{p.stderr.decode('utf-8', 'ignore')}"
        )

    return json.loads(p.stdout.decode("utf-8"))


def make_mp4_animation(spike_t, spike_i, NE, duration_s, fps=15):
    """
    Create a simple 3D scatter "flash on spike" MP4.
    Requires ffmpeg available on the host.
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    rng = np.random.default_rng(0)
    pos = rng.uniform(-1, 1, size=(NE, 3))
    pos[:, 2] *= 0.6

    spike_t = np.asarray(spike_t, dtype=float)
    spike_i = np.asarray(spike_i, dtype=int)

    n_frames = int(duration_s * fps)
    spikes_by_frame = [[] for _ in range(n_frames)]
    if len(spike_t) > 0:
        f_idx = np.clip((spike_t * fps).astype(int), 0, n_frames - 1)
        for f, idx in zip(f_idx, spike_i):
            if 0 <= idx < NE:
                spikes_by_frame[f].append(idx)

    base_size, flash_size, fade = 8.0, 80.0, 0.85
    base_col = np.array([0.2, 0.7, 1.0, 0.12])
    flash_col = np.array([1.0, 0.2, 0.2, 1.0])

    sizes = np.full(NE, base_size, dtype=float)
    cols = np.tile(base_col, (NE, 1))

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], s=sizes, c=cols)

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_zlim(-1.0, 1.0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    title = ax.set_title("t = 0.00 s")

    def update(frame):
        nonlocal sizes, cols
        sizes[:] = np.maximum(base_size, sizes * fade)
        cols[:] = cols * fade + base_col * (1.0 - fade)

        idxs = spikes_by_frame[frame]
        if idxs:
            sizes[idxs] = flash_size
            cols[idxs] = flash_col

        sc._sizes = sizes
        sc.set_color(cols)
        title.set_text(f"t = {frame/fps:.2f} s")
        return sc, title

    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000 / fps, blit=False)

    tmpdir = tempfile.mkdtemp()
    outpath = os.path.join(tmpdir, "neurons.mp4")
    anim.save(outpath, writer="ffmpeg", fps=fps, dpi=160)
    plt.close(fig)
    return outpath


# ---------------------------
# UI
# ---------------------------

uploaded = st.file_uploader("Upload audio
