# Mnemos — Neural Spiking Visualizer (Streamlit + Brian2)

Mnemos is an interactive app that converts an uploaded audio clip into a sparse sequence of rhythmic **drive events** (onsets or beats), drives a simple excitatory/inhibitory (E/I) spiking neural network simulated with **Brian2**, and visualizes the resulting activity as spectrogram overlays, spike rasters, population rates, a membrane trace, and a synced browser animation.

---

## Demo overview

**Pipeline**
1. Upload audio (wav/mp3/ogg/flac/m4a).
2. Extract drive events:
   - **Onsets**: transient attack points.
   - **Beats**: tempo-consistent beat pulses.
3. Run Brian2 network simulation (executed in a separate Python process).
4. Visualize:
   - Spectrogram + drive/spike overlays
   - E/I spike rasters
   - E/I population firing rates
   - Example membrane potential trace
   - Animated “neuron dots” + synapse lines, synced with audio playback

---

## Repository structure (typical)

