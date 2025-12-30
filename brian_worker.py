# brian_worker.py
import numpy as np
from brian2 import *
import json
import sys

def run_sim(drivetimes, T, dt_ms=1.0):
    start_scope()
    defaultclock.dt = dt_ms * ms

    dr = np.asarray(drivetimes, dtype=float)
    indices = np.zeros(len(dr), dtype=int)
    B = SpikeGeneratorGroup(1, indices=indices, times=dr * second)

    tau = 10*ms
    eqs = "dv/dt = -v/tau : 1 (unless refractory)"
    NE, NI = 80, 20
    E = NeuronGroup(NE, eqs, threshold="v>1", reset="v=0", refractory=10*ms, method="exact")
    I = NeuronGroup(NI, eqs, threshold="v>1", reset="v=0", refractory=10*ms, method="exact")
    E.v = "rand()*0.5"
    I.v = "rand()*0.5"

    Sbe = Synapses(B, E, on_pre="v_post += 1.2"); Sbe.connect()
    Sei = Synapses(E, I, on_pre="v_post += 0.5"); Sei.connect(p=0.2)
    Sie = Synapses(I, E, on_pre="v_post += -0.6"); Sie.connect(p=0.2)
    See = Synapses(E, E, on_pre="v_post += 0.05"); See.connect(p=0.05)

    spE = SpikeMonitor(E)
    rateE = PopulationRateMonitor(E)

    net = Network(B, E, I, Sbe, Sei, Sie, See, spE, rateE)
    net.run(T * second)

    return {
        "NE": int(NE),
        "spE_t": (np.array(spE.t/second)).tolist(),
        "spE_i": (np.array(spE.i)).tolist(),
        "rateE_t": (np.array(rateE.t/second)).tolist(),
        "rateE_hz": (np.array(rateE.smooth_rate(window="flat", width=50*ms))).tolist(),
    }

if __name__ == "__main__":
    payload = json.loads(sys.stdin.read())
    out = run_sim(payload["drivetimes"], payload["T"], payload.get("dt_ms", 1.0))
    print(json.dumps(out))

