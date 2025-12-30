# Mnemos

Mnemos is a "Brain DJ": a spiking neural network (Brian2) that listens to a song's beat and generates a spike-driven drum/click track.

## MVP (current goal)
Audio -> beat times -> SpikeGeneratorGroup -> spiking E/I circuit -> spike times -> click-track WAV + raster plot.

## Quickstart
pip install -r requirements.txt
python src/sim/00_sanity_spikegen.py

## Outputs
- outputs/samples/00_sanity_spikegen.png

