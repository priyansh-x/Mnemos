
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
