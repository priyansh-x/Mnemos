import io
import os
import tempfile

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import librosa
import librosa.display

from brian2 import (
    start_scope, defaultclock, ms, second,
    SpikeGeneratorGroup, NeuronGroup, Synapses,
    SpikeMonitor, PopulationRateMonitor, StateMonitor,
    Network
)

st.set_page_config(page_title="Mnemos", layout="wide")
st.title("Mnemos: Music → Beats/Onsets → Spiking E/I Network")

MAX_SECONDS = 30.0
DEFAULT_DT_MS = 1.0

# ---------- UI ----------
uploaded = st.file_uploader("Upload audio (WAV recommended)", type=["wav"])
if uploaded is None:
    st.info("Upload a WAV file to start.")
    st.stop()

audio_bytes = uploaded.getvalue()
st.audio(audio_bytes, format="audio/wav")

# ---------- Audio decode ----------
with st.spinner("Loading audio..."):
    y, sr = librosa.load(io.BytesIO(audio_bytes), sr=None, mono=True, duration=MAX_SECONDS)

T = min(MAX_SECONDS, len(y) / sr)
yT = y[: int(T * sr)]

st.caption(f"Using first {T:.2f} seconds @ {sr} Hz")

# ---------- Beats / onsets ----------
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

# ---------- Spectrogram plot ----------
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

# ---------- Brian2 simulation (cached) ----------
@st.cache_data(show_spinner=False)
def run_brian_ei(drivetimes_sec: np.ndarray, duration_sec: float, dt_ms: float):
    start_scope()
    defaultclock.dt = dt_ms * ms

    dr = np.asarray(drivetimes_sec, dtype=float)
    dr = dr[(dr >= 0.0) & (dr <= duration_sec)]
    dr = np.unique(dr)

    indices = np.zeros(len(dr), dtype=int)
    B = SpikeGeneratorGroup(1, indices=indices, times=dr * second)

    tau = 10 * ms
    eqs = "dv/dt = -v/tau : 1 (unless refractory)"

    NE, NI = 80, 20
    E = NeuronGroup(NE, eqs, threshold="v>1", reset="v=0", refractory=10*ms, method="exact")
    I = NeuronGroup(NI, eqs, threshold="v>1", reset="v=0", refractory=10*ms, method="exact")

    E.v = "rand()*0.5"
    I.v = "rand()*0.5"

    # Synapses (same structure as your notebook) [file:635]
    Sbe = Synapses(B, E, on_pre="v_post += 1.2")
    Sbe.connect()

    Sei = Synapses(E, I, on_pre="v_post += 0.5")
    Sei.connect(p=0.2)

    Sie = Synapses(I, E, on_pre="v_post += -0.6")
    Sie.connect(p=0.2)

    See = Synapses(E, E, on_pre="v_post += 0.05")
    See.connect(p=0.05)

    spE = SpikeMonitor(E)
    spI = SpikeMonitor(I)
    spB = SpikeMonitor(B)

    rateE = PopulationRateMonitor(E)
    rateI = PopulationRateMonitor(I)

    Mv = StateMonitor(E, "v", record=0)

    net = Network(B, E, I, Sbe, Sei, Sie, See, spE, spI, spB, rateE, rateI, Mv)
    net.run(duration_sec * second)

    # Return only numpy-serializable things for Streamlit cache
    out = {
        "NE": NE,
        "NI": NI,
        "drivetimes": dr,
        "spE_t": np.array(spE.t/second),
        "spE_i": np.array(spE.i),
        "spI_t": np.array(spI.t/second),
        "spI_i": np.array(spI.i),
        "rateE_t": np.array(rateE.t/second),
        "rateE_hz": np.array(rateE.smooth_rate(window="flat", width=50*ms)/1.0),  # Hz-like
        "rateI_t": np.array(rateI.t/second),
        "rateI_hz": np.array(rateI.smooth_rate(window="flat", width=50*ms)/1.0),
        "Mv_t": np.array(Mv.t/second),
        "Mv_v0": np.array(Mv.v[0]),
        "syn_Sbe": int(len(Sbe.i)),
        "syn_Sei": int(len(Sei.i)),
        "syn_Sie": int(len(Sie.i)),
        "syn_See": int(len(See.i)),
    }
    return out

run_btn = st.button("Run spiking simulation")
if not run_btn:
    st.stop()

with st.spinner("Running Brian2 E/I network..."):
    sim = run_brian_ei(drivetimes, T, DEFAULT_DT_MS)

with colB:
    st.subheader("Simulation stats")
    st.write(f"Input events: {len(sim['drivetimes'])}")
    st.write(f"Total E spikes: {len(sim['spE_t'])}")
    st.write(f"Total I spikes: {len(sim['spI_t'])}")
    st.write(f"Sbe synapses: {sim['syn_Sbe']}, Sei: {sim['syn_Sei']}, Sie: {sim['syn_Sie']}, See: {sim['syn_See']}")

# ---------- Plots ----------
st.subheader("Neural activity")

fig1, ax1 = plt.subplots(figsize=(12, 3))
ax1.plot(sim["rateE_t"], sim["rateE_hz"], label="E rate")
ax1.plot(sim["rateI_t"], sim["rateI_hz"], label="I rate")
ax1.set_xlabel("time (s)")
ax1.set_ylabel("rate (a.u.)")
ax1.legend()
st.pyplot(fig1, clear_figure=True)

fig2, ax2 = plt.subplots(figsize=(12, 3))
ax2.plot(sim["spE_t"], sim["spE_i"], ".k", markersize=2)
ax2.set_xlabel("time (s)")
ax2.set_ylabel("E neuron index")
st.pyplot(fig2, clear_figure=True)

fig3, ax3 = plt.subplots(figsize=(12, 2.5))
ax3.plot(sim["Mv_t"], sim["Mv_v0"])
ax3.set_xlabel("time (s)")
ax3.set_ylabel("E[0] membrane v")
st.pyplot(fig3, clear_figure=True)

# ---------- Animation: save MP4 and show ----------
st.subheader("3D spike animation (video)")

def make_mp4_animation(spike_t, spike_i, NE, duration_s, fps=15):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    rng = np.random.default_rng(0)
    pos = rng.uniform(-1, 1, size=(NE, 3))
    pos[:, 2] *= 0.6

    n_frames = int(duration_s * fps)
    spikes_by_frame = [[] for _ in range(n_frames)]
    if len(spike_t) > 0:
        f_idx = np.clip((spike_t * fps).astype(int), 0, n_frames - 1)
        for f, idx in zip(f_idx, spike_i.astype(int)):
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
    anim.save(outpath, writer="ffmpeg", fps=fps, dpi=160)  # needs ffmpeg installed
    plt.close(fig)
    return outpath

with st.spinner("Rendering MP4 animation (requires ffmpeg on host)..."):
    try:
        mp4_path = make_mp4_animation(
            sim["spE_t"], sim["spE_i"], sim["NE"], duration_s=min(T, 12.0), fps=15
        )
        with open(mp4_path, "rb") as f:
            st.video(f.read())
        st.caption("Showing first ~12 seconds for faster render.")
    except Exception as e:
        st.error("MP4 rendering failed (likely missing ffmpeg on the server).")
        st.code(str(e))
        st.info("Fallback: keep only plots, or deploy with a host that has ffmpeg available.")


