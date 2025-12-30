# Mnemos — Neural Spiking Visualizer (Streamlit + Brian2)

Mnemos is an interactive application that converts an uploaded audio clip into a sparse sequence of rhythmic **drive events** (onsets or beats), drives a simple excitatory/inhibitory (E/I) spiking neural network simulated with **Brian2**, and visualizes the resulting activity through spectrogram overlays, spike rasters, population rates, a membrane trace, and a synced browser animation.

---

## What Mnemos does

### Input
- Upload an audio file (wav, mp3, ogg, flac, m4a).

### Event extraction (“drive”)
- **Onsets mode**: detects transient/attack points in the audio (useful for percussive signals).
- **Beats mode**: estimates tempo and emits beat times (useful for rhythmic tracks).

These timestamps become the “input spike train” that stimulates the spiking network.

### Spiking simulation
- A minimal E/I spiking network is run using **Brian2**.
- The simulation is launched in a **separate subprocess** to avoid instability that can occur when running Brian2 directly inside Streamlit’s rerun/thread model.
- Outputs are saved to a temporary `.npz` file and loaded back into the Streamlit process for plotting.

### Visualization outputs
- Spectrogram of the chosen time window with:
  - drive event overlay (blue)
  - excitatory spike overlay (red)
- Spike rasters:
  - E population raster
  - I population raster
- Population rate plots:
  - E population rate
  - I population rate
- Example membrane potential trace (one recorded E neuron)
- Animation:
  - neuron dots “flash” when they spike
  - synapse lines can be drawn between nodes (sampled for performance)
  - Play starts both animation and audio; Reset seeks both back to time 0

---

## Repository layout (typical)

```
.
├─ app.py                   # Streamlit app entrypoint (main file path for Streamlit)
├─ simulate_brian2.py        # Brian2 subprocess script (may be auto-generated/updated)
├─ brian_worker.py           # (optional) worker helper (if present)
├─ 01_sanity_spikegen.ipynb  # notebook reference / sanity checks
├─ requirements.txt
├─ packages.txt              # (optional) OS-level packages (Streamlit Cloud)
├─ LICENSE
├─ README.md
├─ configs/                  # (optional)
└─ src/                      # (optional)
```

If Streamlit asks for “Main file path”, use:
- `app.py`

---

## Installation

### Option A — recommended: virtual environment + requirements.txt

**macOS/Linux**
```
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

**Windows (PowerShell)**
```
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -U pip
pip install -r requirements.txt
```

### Option B — minimal manual install
```
pip install streamlit numpy matplotlib librosa brian2
```

---

## Run locally

```
streamlit run app.py
```

Then open the URL printed in the terminal (often `http://localhost:8501`).

---

## How the Brian2 network is structured (high-level)

Mnemos uses an event-driven input and a basic recurrent E/I network:

- **Input**: SpikeGeneratorGroup with spike times = extracted drive events
- **Excitatory neurons (E)**: leaky integrate-and-fire style dynamics (simple decay + threshold/reset)
- **Inhibitory neurons (I)**: same style, used to provide negative feedback
- **Synapses** (typical):
  - Input → E (B→E)
  - E → I
  - I → E (negative weight)
  - E → E (sparse recurrent)

Key knobs exposed in the UI:
- `dt (ms)`, `NE`, `NI`, `tau`, `refractory`
- synaptic weights and connection probabilities

---

## Usage guide

### Step-by-step
1. Upload audio.
2. Choose drive extraction mode: `onsets` or `beats`.
3. Choose simulation duration `T` and animation FPS.
4. Tune network parameters if desired.
5. Click **Run simulation**.
6. Use the **Play** button near the animation to start both audio + animation.
7. Use **Reset** to set both back to t = 0.

---

## Performance notes

- Larger `NE`/`NI` and smaller `dt` can increase simulation time.
- Higher animation FPS increases browser load and the size of the embedded animation payload.
- Drawing synapse lines is visually helpful but can be heavy; sampling the number of edges keeps it responsive.

---

## Troubleshooting

### “Brian2 subprocess failed”
- Verify your environment has Brian2 installed:
```
python -c "import brian2; print(brian2.__version__)"
```
- Ensure `simulate_brian2.py` matches the CLI arguments your `app.py` is using (especially if the app auto-generates/overwrites it).

### “No drive events found”
- Switch `onsets` ↔ `beats`
- Increase `T`
- Use a more rhythmic/percussive audio section

### Animation looks cut off or unstable
- Reduce FPS
- Reduce NE/NI
- Reduce number of synapse lines (edge sampling)
- If you recently changed T/fps, re-run the simulation so the embedded animation regenerates for the new duration.

---

## Deployment (Streamlit Community Cloud)

1. Push repo to GitHub.
2. Create a Streamlit app:
   - **Main file path**: `app.py`
3. Ensure `requirements.txt` exists (and `packages.txt` if needed).

---

## Roadmap ideas

- Multi-band audio preprocessing → multi-channel spike drives
- Add plasticity rules (e.g., STDP)
- Export results (spike trains, rates) to CSV
- Replace Matplotlib JS animation with a WebGL-first renderer for long-duration + smooth playback

---

## License

See `LICENSE`.
```

This uses an outer fence of four backticks so the inner triple-backtick blocks (like the repository tree and bash commands) stay inside the same big “text box”.[1][2]
