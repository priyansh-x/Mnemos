# app.py
import os
import io
import sys
import tempfile
import subprocess
import base64

import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import librosa
import librosa.display


# ----------------------------
# Matplotlib: allow larger JS-embedded animations
# ----------------------------
# Matplotlib's JS/HTML embedding has an embed size limit (MB). If exceeded,
# later frames may not be embedded -> animation appears to stop early. [web:93]
mpl.rcParams["animation.embed_limit"] = 200.0  # MB


# =============================================================================
# Mnemos — Neural Spiking Visualizer
# =============================================================================
st.set_page_config(page_title="Mnemos", layout="wide")
st.title("Mnemos — Neural Spiking Visualizer")
st.caption("Audio → events → spiking network → interpretable visualizations + synced animation.")

with st.expander("What is Mnemos?", expanded=True):
    st.markdown(
        """
Mnemos is an interactive demo that converts an **audio file** into a stream of time-stamped events (either *onsets* or *beats*),
uses those events as an input stimulus to a small **spiking neural network**, and then visualizes how activity propagates through
the network over time.

### What it does
- Extracts a sparse “event train” from audio (onset or beat times).
- Drives a Brian2 E/I network (Excitatory + Inhibitory populations) using those events.
- Displays:
  - Spectrogram with event and spike overlays
  - Spike rasters (E and I)
  - Population firing rates (E and I)
  - Example membrane trace
  - Browser-rendered animation synced with the audio

### Why it’s useful
- Builds intuition for event-driven spiking systems.
- Lets you tune synaptic weights/probabilities and see spiking change instantly.
- Demonstrates a compact “audio → spikes → viz” pipeline.
        """.strip()
    )


# ----------------------------
# Brian2 subprocess helper file
# ----------------------------
SIM_SCRIPT_NAME = "simulate_brian2.py"

SIM_SCRIPT_CONTENT = r"""
import argparse
import numpy as np

def main():
    from brian2 import (
        start_scope, defaultclock,
        SpikeGeneratorGroup, NeuronGroup, Synapses,
        SpikeMonitor, PopulationRateMonitor, StateMonitor, Network,
        second, ms
    )

    ap = argparse.ArgumentParser()
    ap.add_argument("--npz_out", required=True)
    ap.add_argument("--drivetimes", required=True)  # path to .npy
    ap.add_argument("--T", type=float, required=True)
    ap.add_argument("--dt_ms", type=float, default=1.0)
    ap.add_argument("--NE", type=int, default=80)
    ap.add_argument("--NI", type=int, default=20)
    ap.add_argument("--tau_ms", type=float, default=10.0)
    ap.add_argument("--refr_ms", type=float, default=10.0)

    ap.add_argument("--w_be", type=float, default=1.2)
    ap.add_argument("--w_ei", type=float, default=0.5)
    ap.add_argument("--w_ie", type=float, default=-0.6)
    ap.add_argument("--w_ee", type=float, default=0.05)
    ap.add_argument("--p_ei", type=float, default=0.2)
    ap.add_argument("--p_ie", type=float, default=0.2)
    ap.add_argument("--p_ee", type=float, default=0.05)

    ap.add_argument("--max_edges", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=0)

    args = ap.parse_args()

    drivetimes = np.load(args.drivetimes).astype(float)
    drivetimes = drivetimes[np.isfinite(drivetimes)]
    drivetimes = np.unique(np.sort(drivetimes))
    T = float(args.T)

    start_scope()
    defaultclock.dt = float(args.dt_ms) * ms

    idx = np.zeros(len(drivetimes), dtype=int)
    B = SpikeGeneratorGroup(1, idx, drivetimes * second)

    tau = float(args.tau_ms) * ms
    refr = float(args.refr_ms) * ms
    eqs = "dv/dt = -v/tau : 1"

    E = NeuronGroup(int(args.NE), eqs, threshold="v>1", reset="v=0", refractory=refr, method="exact")
    I = NeuronGroup(int(args.NI), eqs, threshold="v>1", reset="v=0", refractory=refr, method="exact")

    E.v = "rand()*0.5"
    I.v = "rand()*0.5"

    Sbe = Synapses(B, E, on_pre=f"v_post += {float(args.w_be)}")
    Sbe.connect()

    Sei = Synapses(E, I, on_pre=f"v_post += {float(args.w_ei)}")
    Sei.connect(p=float(args.p_ei))

    Sie = Synapses(I, E, on_pre=f"v_post += {float(args.w_ie)}")
    Sie.connect(p=float(args.p_ie))

    See = Synapses(E, E, on_pre=f"v_post += {float(args.w_ee)}")
    See.connect(p=float(args.p_ee))

    spE = SpikeMonitor(E)
    spI = SpikeMonitor(I)
    spB = SpikeMonitor(B)
    rateE = PopulationRateMonitor(E)
    rateI = PopulationRateMonitor(I)
    v0 = StateMonitor(E, "v", record=0)

    net = Network(B, E, I, Sbe, Sei, Sie, See, spB, spE, spI, rateE, rateI, v0)
    net.run(T * second)

    rng = np.random.default_rng(int(args.seed))
    def sample_edges(S):
        pre = np.asarray(S.i[:], dtype=np.int32)
        post = np.asarray(S.j[:], dtype=np.int32)
        if pre.size == 0:
            return pre, post
        m = int(args.max_edges)
        if pre.size > m:
            sel = rng.choice(pre.size, size=m, replace=False)
            pre = pre[sel]
            post = post[sel]
        return pre, post

    See_pre, See_post = sample_edges(See)
    Sei_pre, Sei_post = sample_edges(Sei)
    Sie_pre, Sie_post = sample_edges(Sie)

    np.savez(
        args.npz_out,
        spB_t=np.asarray(spB.t / second),

        tE=np.asarray(spE.t / second),
        iE=np.asarray(spE.i, dtype=np.int32),

        tI=np.asarray(spI.t / second),
        iI=np.asarray(spI.i, dtype=np.int32),

        rateE_t=np.asarray(rateE.t / second),
        rateE=np.asarray(rateE.rate / (1/second)),

        rateI_t=np.asarray(rateI.t / second),
        rateI=np.asarray(rateI.rate / (1/second)),

        v0_t=np.asarray(v0.t / second),
        v0=np.asarray(v0.v[0]),

        See_pre=See_pre, See_post=See_post,
        Sei_pre=Sei_pre, Sei_post=Sei_post,
        Sie_pre=Sie_pre, Sie_post=Sie_post,
    )

if __name__ == "__main__":
    main()
"""

def ensure_sim_script(force=False):
    if force or (not os.path.exists(SIM_SCRIPT_NAME)):
        with open(SIM_SCRIPT_NAME, "w", encoding="utf-8") as f:
            f.write(SIM_SCRIPT_CONTENT)

ensure_sim_script(force=True)


# ----------------------------
# Helper utilities
# ----------------------------
def safe_resample(y, sr_in, sr_out):
    if sr_in == sr_out:
        return y
    return librosa.resample(y, orig_sr=sr_in, target_sr=sr_out)

def compute_drivetimes(y, sr, mode="onsets"):
    if mode == "onsets":
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
        onset_times = librosa.frames_to_time(onset_frames, sr=sr)
        return np.asarray(onset_times, dtype=float)
    else:
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, units="frames")
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        return np.asarray(beat_times, dtype=float)

def plot_spectrogram_with_overlays(y, sr, T, drivetimes, out_spike_times, fmax=12000):
    yT = y[: int(T * sr)]
    D = librosa.stft(yT)
    S = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    fig, ax = plt.subplots(figsize=(12, 4))
    img = librosa.display.specshow(S, x_axis="time", y_axis="hz", sr=sr, ax=ax)
    ax.set_title("Spectrogram + drive events (blue) + E spikes (red)")
    ax.set_ylim(0, fmax)
    ax.set_xlim(0, T)
    fig.colorbar(img, ax=ax, format="%+2.0f dB")

    driveT = drivetimes[(drivetimes >= 0) & (drivetimes <= T)]
    for t in driveT:
        ax.axvline(float(t), color="dodgerblue", linewidth=1.0, alpha=0.55)

    outT = out_spike_times[(out_spike_times >= 0) & (out_spike_times <= T)]
    if len(outT) > 500:
        idx = np.linspace(0, len(outT) - 1, 500).astype(int)
        outT = outT[idx]
    for t in outT:
        ax.axvline(float(t), color="red", linewidth=1.0, alpha=0.35)

    plt.tight_layout()
    return fig

def raster_fig(spike_t, spike_i, T, title, max_points=200000):
    mask = (spike_t >= 0) & (spike_t <= T)
    t = spike_t[mask]
    i = spike_i[mask]
    if len(t) > max_points:
        sel = np.linspace(0, len(t) - 1, max_points).astype(int)
        t, i = t[sel], i[sel]
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.scatter(t, i, s=2, c="k", alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Neuron index")
    ax.set_xlim(0, T)
    plt.tight_layout()
    return fig

def rate_fig(t, r, T, title):
    mask = (t >= 0) & (t <= T)
    fig, ax = plt.subplots(figsize=(12, 2.6))
    ax.plot(t[mask], r[mask], lw=1.4)
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Hz")
    ax.set_xlim(0, T)
    plt.tight_layout()
    return fig

def _pad_or_truncate_audio_bytes_to_T(y_float, sr, T):
    n_target = int(np.round(T * sr))
    if len(y_float) < n_target:
        y2 = np.pad(y_float, (0, n_target - len(y_float)))
    else:
        y2 = y_float[:n_target]

    y16 = np.clip(y2, -1.0, 1.0)
    y16 = (y16 * 32767.0).astype(np.int16)

    import wave
    bio = io.BytesIO()
    with wave.open(bio, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sr))
        wf.writeframes(y16.tobytes())
    return bio.getvalue(), "audio/wav"


def animation_with_audio_js(
    spike_t, spike_i,
    N, T, fps,
    audio_bytes_wav,
    edges,
    drive_times,
    render_nonce
):
    """
    Streamlit components.html has no key= in some versions. [web:168]
    Workaround: bake a changing nonce into the HTML string so Streamlit re-renders it.
    """
    nframes = max(1, int(np.ceil(float(T) * int(fps))))
    N = int(N)

    rng = np.random.default_rng(0)
    pos = rng.uniform(-1, 1, size=(N, 3))
    pos[:, 2] *= 0.6

    spikes_by_frame = [[] for _ in range(nframes)]
    if spike_t is not None and len(spike_t) > 0:
        fidx = np.clip((np.asarray(spike_t) * fps).astype(int), 0, nframes - 1)
        for f, idx in zip(fidx, spike_i):
            ii = int(idx)
            if 0 <= ii < N:
                spikes_by_frame[int(f)].append(ii)

    drive_frames = set()
    if drive_times is not None and len(drive_times) > 0:
        df = np.clip((np.asarray(drive_times) * fps).astype(int), 0, nframes - 1)
        drive_frames = set(df.tolist())

    base_size, flash_size, fade = 8.0, 80.0, 0.85
    base_col = np.array([0.2, 0.7, 1.0, 0.18])
    flash_col = np.array([1.0, 0.2, 0.2, 1.0])

    sizes = np.full(N, base_size, dtype=float)
    cols = np.tile(base_col, (N, 1))

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")

    def draw_edges(pre, post, color, alpha, lw):
        pre = np.asarray(pre, dtype=int)
        post = np.asarray(post, dtype=int)
        m = min(len(pre), len(post))
        for a, b in zip(pre[:m], post[:m]):
            if 0 <= a < N and 0 <= b < N:
                ax.plot(
                    [pos[a, 0], pos[b, 0]],
                    [pos[a, 1], pos[b, 1]],
                    [pos[a, 2], pos[b, 2]],
                    color=color, alpha=alpha, linewidth=lw
                )

    draw_edges(edges["See_pre"], edges["See_post"], color=(0.85, 0.85, 0.85), alpha=0.08, lw=0.6)
    draw_edges(edges["Sei_pre"], edges["Sei_post"], color=(0.2, 1.0, 0.2), alpha=0.06, lw=0.6)
    draw_edges(edges["Sie_pre"], edges["Sie_post"], color=(1.0, 0.4, 0.2), alpha=0.06, lw=0.6)

    sc = ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], s=sizes, c=cols)

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_zlim(-1.0, 1.0)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])

    title = ax.set_title("t = 0.00 s")
    badge = ax.text2D(0.02, 0.95, "", transform=ax.transAxes)

    def update(frame):
        nonlocal sizes, cols
        sizes = np.maximum(base_size, sizes * fade)
        cols = cols * fade + base_col * (1.0 - fade)

        idxs = spikes_by_frame[frame]
        if idxs:
            sizes[idxs] = flash_size
            cols[idxs] = flash_col

        sc.set_sizes(sizes)
        sc.set_color(cols)

        tnow = frame / fps
        title.set_text(f"t = {tnow:.2f} s")
        badge.set_text("Drive!" if frame in drive_frames else "")
        return sc, title, badge

    anim = FuncAnimation(fig, update, frames=nframes, interval=1000 / fps, blit=False)

    # With embed_limit increased above, embed_frames=True should not truncate. [web:93]
    anim_html = anim.to_jshtml(embed_frames=True)

    b64 = base64.b64encode(audio_bytes_wav).decode("utf-8")

    # IDs include nonce so JS binds to the current render, not old iframe DOM.
    play_id = f"play_both_{render_nonce}"
    reset_id = f"reset_both_{render_nonce}"
    audio_id = f"music_{render_nonce}"
    container_id = f"anim_container_{render_nonce}"

    html = f"""
    <!-- nonce forces Streamlit to treat this as new HTML when sliders change -->
    <meta name="mnemos-render-nonce" content="{render_nonce}">

    <div style="display:flex; flex-direction:column; gap:10px;">
      <div style="display:flex; gap:10px; align-items:center;">
        <button id="{play_id}" style="padding:8px 14px; font-size:16px; width:160px;">Play</button>
        <button id="{reset_id}" style="padding:8px 14px; font-size:16px; width:160px;">Reset</button>
        <span style="font-size:14px; opacity:0.8;">Play starts audio + animation. Reset puts both back to t=0.</span>
      </div>

      <audio id="{audio_id}" controls style="width: 100%;">
        <source src="data:audio/wav;base64,{b64}" type="audio/wav">
      </audio>

      <div id="{container_id}">
        {anim_html}
      </div>
    </div>

    <script>
      function q(sel) {{ return document.querySelector(sel); }}
      function findBtn(container, titleText) {{
        if (!container) return null;
        return container.querySelector(`button[title="${{titleText}}"]`)
            || container.querySelector(`button[aria-label="${{titleText}}"]`);
      }}

      async function playBoth() {{
        const audio = q("#{audio_id}");
        const container = q("#{container_id}");
        const playBtn = findBtn(container, "Play");
        if (audio) {{
          try {{ await audio.play(); }} catch (e) {{ console.log("Audio play blocked:", e); }}
        }}
        if (playBtn) playBtn.click();
      }}

      function resetBoth() {{
        const audio = q("#{audio_id}");
        const container = q("#{container_id}");
        const pauseBtn = findBtn(container, "Pause");
        const firstBtn = findBtn(container, "First frame");
        if (pauseBtn) pauseBtn.click();
        if (firstBtn) firstBtn.click();
        if (audio) {{
          audio.pause();
          audio.currentTime = 0;
        }}
      }}

      const playButton = q("#{play_id}");
      const resetButton = q("#{reset_id}");
      if (playButton) playButton.addEventListener("click", playBoth);
      if (resetButton) resetButton.addEventListener("click", resetBoth);
    </script>
    """

    components.html(html, height=800, scrolling=True)
    plt.close(fig)


# ----------------------------
# Sidebar UI
# ----------------------------
with st.sidebar:
    st.header("Audio")
    audio_file = st.file_uploader("Upload audio", type=["wav", "mp3", "ogg", "flac", "m4a"])

    st.header("Drive extraction")
    drive_mode = st.selectbox("Drive events", ["onsets", "beats"], index=0)
    target_sr = st.select_slider("Processing sample rate", options=[8000, 11025, 16000, 22050, 44100], value=22050)

    st.header("Simulation window")
    T = st.slider("Sim duration T (s)", 1.0, 60.0, 20.0, 0.5)
    fps = st.slider("Animation FPS", 5, 60, 30, 1)

    st.header("Brian2 params")
    dt_ms = st.slider("dt (ms)", 0.1, 5.0, 1.0, 0.1)
    NE = st.slider("E neurons", 10, 400, 80, 10)
    NI = st.slider("I neurons", 5, 200, 20, 5)
    tau_ms = st.slider("tau (ms)", 1.0, 50.0, 10.0, 1.0)
    refr_ms = st.slider("refractory (ms)", 1.0, 50.0, 10.0, 1.0)

    st.header("Synapses")
    w_be = st.slider("Input→E weight", 0.0, 5.0, 1.2, 0.05)
    w_ei = st.slider("E→I weight", 0.0, 2.0, 0.5, 0.05)
    w_ie = st.slider("I→E weight (negative)", -2.0, 0.0, -0.6, 0.05)
    w_ee = st.slider("E→E weight", 0.0, 0.5, 0.05, 0.01)

    p_ei = st.slider("E→I connect p", 0.0, 1.0, 0.2, 0.05)
    p_ie = st.slider("I→E connect p", 0.0, 1.0, 0.2, 0.05)
    p_ee = st.slider("E→E connect p", 0.0, 1.0, 0.05, 0.01)

    run_btn = st.button("Run simulation")


# ----------------------------
# Main flow
# ----------------------------
if audio_file is None:
    st.info("Upload an audio file to start.")
    st.stop()

audio_bytes_in = audio_file.read()

y, sr0 = librosa.load(io.BytesIO(audio_bytes_in), sr=None, mono=True)
y = safe_resample(y, sr0, target_sr)
sr = target_sr

T_target = float(T)

n_target = int(np.round(T_target * sr))
if len(y) < n_target:
    y_clip = np.pad(y, (0, n_target - len(y)))
else:
    y_clip = y[:n_target]

drivetimes = compute_drivetimes(y_clip, sr, mode=drive_mode)
drivetimes = drivetimes[np.isfinite(drivetimes)]
drivetimes = drivetimes[(drivetimes >= 0) & (drivetimes <= T_target)]
drivetimes = np.unique(np.sort(drivetimes))

st.write(f"Detected **{len(drivetimes)}** drive events using **{drive_mode}**.")

with st.expander("Input spectrogram", expanded=True):
    fig, ax = plt.subplots(figsize=(12, 4))
    D = librosa.stft(y_clip)
    S = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    img = librosa.display.specshow(S, x_axis="time", y_axis="hz", sr=sr, ax=ax)
    ax.set_title("Input Spectrogram")
    ax.set_xlim(0, T_target)
    ax.set_ylim(0, 12000)
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

if not run_btn:
    st.stop()

if len(drivetimes) == 0:
    st.error("No drive events found. Try 'beats', increase T, or use a different segment.")
    st.stop()

with st.spinner("Running Brian2 simulation in subprocess..."):
    with tempfile.TemporaryDirectory() as td:
        drive_path = os.path.join(td, "drivetimes.npy")
        out_path = os.path.join(td, "out.npz")
        np.save(drive_path, drivetimes.astype(float))

        cmd = [
            sys.executable, SIM_SCRIPT_NAME,
            "--npz_out", out_path,
            "--drivetimes", drive_path,
            "--T", str(float(T_target)),
            "--dt_ms", str(float(dt_ms)),
            "--NE", str(int(NE)),
            "--NI", str(int(NI)),
            "--tau_ms", str(float(tau_ms)),
            "--refr_ms", str(float(refr_ms)),
            "--w_be", str(float(w_be)),
            "--w_ei", str(float(w_ei)),
            "--w_ie", str(float(w_ie)),
            "--w_ee", str(float(w_ee)),
            "--p_ei", str(float(p_ei)),
            "--p_ie", str(float(p_ie)),
            "--p_ee", str(float(p_ee)),
            "--max_edges", "1200",
            "--seed", "0",
        ]

        try:
            subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            st.error("Brian2 subprocess failed.")
            st.code(e.output.decode("utf-8", errors="ignore"))
            st.stop()

        data = np.load(out_path)

tE, iE = data["tE"], data["iE"]
tI, iI = data["tI"], data["iI"]
rateE_t, rateE = data["rateE_t"], data["rateE"]
rateI_t, rateI = data["rateI_t"], data["rateI"]
v0_t, v0 = data["v0_t"], data["v0"]

edges = {
    "See_pre": data["See_pre"],
    "See_post": data["See_post"],
    "Sei_pre": data["Sei_pre"],
    "Sei_post": data["Sei_post"],
    "Sie_pre": data["Sie_pre"],
    "Sie_post": data["Sie_post"],
}

st.success("Simulation complete.")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Drive events", int(len(drivetimes)))
c2.metric("E spikes", int(len(tE)))
c3.metric("I spikes", int(len(tI)))
c4.metric("dt (ms)", float(dt_ms))

st.subheader("Spectrogram + drive + spikes")
fig_spec = plot_spectrogram_with_overlays(y_clip, sr, T_target, drivetimes, tE)
st.pyplot(fig_spec)
plt.close(fig_spec)

st.subheader("Population rates")
fig_re = rate_fig(rateE_t, rateE, T_target, "E population rate")
st.pyplot(fig_re)
plt.close(fig_re)
fig_ri = rate_fig(rateI_t, rateI, T_target, "I population rate")
st.pyplot(fig_ri)
plt.close(fig_ri)

st.subheader("Spike rasters")
fig_e = raster_fig(tE, iE, T_target, f"E raster (NE={NE})")
st.pyplot(fig_e)
plt.close(fig_e)
fig_i = raster_fig(tI, iI, T_target, f"I raster (NI={NI})")
st.pyplot(fig_i)
plt.close(fig_i)

st.subheader("Example membrane trace (E[0])")
maskv = (v0_t >= 0) & (v0_t <= T_target)
figv, axv = plt.subplots(figsize=(12, 2.6))
axv.plot(v0_t[maskv], v0[maskv])
axv.set_xlabel("Time (s)")
axv.set_ylabel("v")
axv.set_title("E[0] membrane potential")
axv.set_xlim(0, T_target)
plt.tight_layout()
st.pyplot(figv)
plt.close(figv)

audio_bytes_playback_wav, _ = _pad_or_truncate_audio_bytes_to_T(y, sr, T_target)

st.subheader("Spiking Animation + Audio (Play/Reset) + Synapse lines")

# This nonce forces a real re-render even without components.html(key=...) support. [web:168]
render_nonce = f"T{int(T_target*1000)}_fps{int(fps)}_NE{int(NE)}_NI{int(NI)}_dt{int(float(dt_ms)*1000)}"

animation_with_audio_js(
    spike_t=tE, spike_i=iE,
    N=int(NE), T=float(T_target), fps=int(fps),
    audio_bytes_wav=audio_bytes_playback_wav,
    edges=edges,
    drive_times=drivetimes,
    render_nonce=render_nonce
)

st.caption("Early stopping was due to embedded-frame truncation; raising animation.embed_limit prevents that. [web:93]")
