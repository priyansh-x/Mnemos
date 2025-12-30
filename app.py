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
    """Runs Brian2 simulation in a separate Python process to avoid Streamlit thread/signal issues."""
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

    ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5); ax.set_zlim(-1.0, 1.0)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
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

    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000/fps, blit=False)

    tmpdir = tempfile.mkdtemp()
    outpath = os.path.join(tmpdir, "neurons.mp4")
    anim.save(outpath, writer="ffmpeg", fps=fps, dpi=160)
    plt.close(fig)
    return outpath


# ---------------------------
# UI
# ---------------------------

uploaded = st.file_uploader("Upload audio (WAV recommended)", type=["wav"])
if uploaded is None:
    st.info("Upload a WAV file to start.")
    st.stop()

audio_bytes = uploaded.getvalue()
st.audio(audio_bytes, format="audio/wav")  # Streamlit supports bytes here [web:524]

# ---------------------------
# Audio processing
# ---------------------------

with st.spinner("Loading audio..."):
    y, sr = librosa.load(io.BytesIO(audio_bytes), sr=None, mono=True, duration=MAX_SECONDS)

T = min(MAX_SECONDS, len(y) / sr)
yT = y[: int(T * sr)]

st.caption(f"Using first {T:.2f} seconds @ {sr} Hz")

with st.spinner("Extracting beat/onset times..."):
    tempo, beat_times = librosa.beat.beat_track(y=yT, sr=sr, units="time")
    onset_frames = librosa.onset.onset_detect(y=yT, sr=sr)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)

beat_times = np.asarray(beat_times, dtype=float)
onset_times = np.asarray(onset_times, dtype=float)

drive_mode = st.radio("Drive spikes from", ["Onsets (recommended)", "Beats"], index=0)
drivetimes = onset_times if drive_mode.startswith("Onsets") else beat_times
drivetimes = drivetimes[(drivetimes >= 0.0) & (drivetimes <= T)]
drivetimes = np.unique(drivetimes)

st.write(f"Tempo estimate: {tempo:.2f} BPM")
st.write(f"Drive events in first {T:.1f}s: {len(drivetimes)}")

# ---------------------------
# Plots: Spectrogram
# ---------------------------

colA, colB = st.columns([1, 1])

with colA:
    st.subheader("Spectrogram + drive events")
    fig, ax = plt.subplots(figsize=(10, 4))
    D = librosa.stft(yT)
    S = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    librosa.display.specshow(S, x_axis="time", y_axis="hz", sr=sr, ax=ax)
    for t0 in drivetimes:
        ax.axvline(t0, color="cyan", linewidth=1.0, alpha=0.5)
    ax.set_xlim(0, T)
    ax.set_ylim(0, 12000)
    ax.set_title("Spectrogram (cyan = drive events)")
    st.pyplot(fig, clear_figure=True)

# ---------------------------
# Simulation
# ---------------------------

run_btn = st.button("Run spiking simulation")
if not run_btn:
    st.stop()

with st.spinner("Running Brian2 in subprocess..."):
    sim = run_brian_subprocess(drivetimes, T, DEFAULT_DT_MS)

# Convert lists back to numpy for plotting
NE = int(sim["NE"])
spE_t = np.asarray(sim["spE_t"], dtype=float)
spE_i = np.asarray(sim["spE_i"], dtype=int)
rateE_t = np.asarray(sim["rateE_t"], dtype=float)
rateE_hz = np.asarray(sim["rateE_hz"], dtype=float)

with colB:
    st.subheader("Simulation stats")
    st.write(f"Input events: {len(drivetimes)}")
    st.write(f"Total E spikes: {len(spE_t)}")
    st.write(f"NE: {NE}")

st.subheader("Neural activity")

fig1, ax1 = plt.subplots(figsize=(12, 3))
ax1.plot(rateE_t, rateE_hz, label="E rate")
ax1.set_xlabel("time (s)")
ax1.set_ylabel("rate (a.u.)")
ax1.legend()
st.pyplot(fig1, clear_figure=True)

fig2, ax2 = plt.subplots(figsize=(12, 3))
ax2.plot(spE_t, spE_i, ".k", markersize=2)
ax2.set_xlabel("time (s)")
ax2.set_ylabel("E neuron index")
st.pyplot(fig2, clear_figure=True)

# ---------------------------
# Animation (optional)
# ---------------------------

st.subheader("Spike animation (video)")

anim_seconds = st.slider("Animation length (seconds)", 5.0, float(min(T, 30.0)), float(min(T, 12.0)), 1.0)
fps = st.slider("FPS", 10, 30, 15, 1)

with st.spinner("Rendering MP4 (requires ffmpeg)..."):
    try:
        mp4_path = make_mp4_animation(spE_t, spE_i, NE, duration_s=anim_seconds, fps=fps)
        with open(mp4_path, "rb") as f:
            st.video(f.read())
        st.caption("If this fails on Streamlit Cloud, add `packages.txt` with `ffmpeg`.")
    except Exception as e:
        st.error("MP4 rendering failed (likely missing ffmpeg on the host).")
        st.code(str(e))
