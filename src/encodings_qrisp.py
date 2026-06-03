"""
Quantum Encoding Circuits — Converted from PennyLane to Qrisp
==============================================================
Original: PennyLane + @qml.qnode decorators
Converted: Qrisp QuantumVariable + functional gate API

Key API mappings:
  PennyLane                    →  Qrisp
  ─────────────────────────────────────────────────────────────
  qml.RX(θ, wires=i)          →  rx(θ, qv[i])
  qml.RY(θ, wires=i)          →  ry(θ, qv[i])
  qml.RZ(θ, wires=i)          →  rz(θ, qv[i])
  qml.Hadamard(wires=i)        →  h(qv[i])
  qml.CZ(wires=[i, j])         →  cz(qv[i], qv[j])
  qml.PhaseShift(θ, wires=i)   →  p(θ, qv[i])
  qml.BasisState(b, wires=…)   →  manual x() on |1⟩ qubits
  qml.AmplitudeEmbedding(…)    →  qs.statevector_sim (see note)
  qml.expval(qml.PauliZ(0))    →  multi_measurement / get_measurement
  @qml.qnode(dev)              →  plain Python function returning qv
  measure_all(N)               →  multi_measurement(qv) or qv.get_measurement()

Notes on AmplitudeEmbedding:
  Qrisp does not have a built-in AmplitudeEmbedding gate.
  The cleanest equivalent is to initialise via QuantumVariable.init_sv()
  (statevector init) which is available in the simulator path, or to
  decompose into uniformly-controlled rotations.  Here we use
  qs.init_state(x_padded) as a semantic placeholder that works with the
  Qrisp statevector simulator.

References:
  Qrisp docs  : https://qrisp.eu/reference/index.html
  PennyLane   : https://pennylane.ai
  Schuld & Killoran, PRL 2022
  Pérez-Salinas et al., Quantum 2020
  Havlíček et al., Nature 2019
  Farhi, Goldstone & Gutmann 2014
  Derouich et al., 2025
"""

import numpy as np
from qrisp import (
    QuantumVariable,
    rx, ry, rz,
    h, cz, x, p,
    multi_measurement,
)

# ── helpers ──────────────────────────────────────────────────────────────────

def pad_amplitude(vec, n_qubits):
    """Zero-pad and L2-normalise vec to length 2**n_qubits."""
    size = 2 ** n_qubits
    out  = np.zeros(size)
    out[:len(vec)] = vec[:size]
    norm = np.linalg.norm(out)
    return out / norm if norm > 0 else out

def bits(n, width):
    """Return list of bits (MSB first) for integer n."""
    return [(n >> (width - 1 - i)) & 1 for i in range(width)]

def variational_layer(qv, weights, n_q, layer):
    """
    Generic variational layer shared by all encodings.
    Mirrors the original variational_layer(weights, N_Q, l):
      - RY(w[l,i,0]) + RZ(w[l,i,1]) on each qubit
      - Ladder of CZ entanglers
    """
    for i in range(n_q):
        ry(weights[layer, i, 0], qv[i])
        rz(weights[layer, i, 1], qv[i])
    for i in range(n_q - 1):
        cz(qv[i], qv[i + 1])

def measure_all(qv):
    """Return measurement dict over all qubits (mirrors measure_all(N))."""
    return qv.get_measurement()

# ── Configuration ─────────────────────────────────────────────────────────────
CFG = {"n_features": 8, "n_layers": 2}
N_Q   = CFG["n_features"]   # 8 qubits
L     = CFG["n_layers"]     # 2 variational layers
N_Q_AMP   = 4               # 2^4 = 16 ≥ 8 features
N_Q_BASIS = 3               # 2^3 = 8 binary states


# ================================================================
# ENCODING 1: ANGLE ENCODING — RX
# ================================================================
# Maps feature xᵢ → RX(xᵢ) on qubit i.
# Pros: Simple, hardware-efficient, depth 1.
# Cons: Limited expressibility (one rotation per qubit).
# Ref: Schuld & Killoran, PRL 2022
# ================================================================

def enc_angle_rx(x_feat, weights):
    qv = QuantumVariable(N_Q)
    # ENCODING
    for i in range(N_Q):
        rx(x_feat[i], qv[i])
    # VARIATIONAL
    for l in range(L):
        variational_layer(qv, weights, N_Q, l)
    return measure_all(qv)


# ================================================================
# ENCODING 2: ANGLE ENCODING — RY+RZ (double rotation)
# ================================================================
# Two rotations per qubit → richer expressibility in the Bloch sphere.
# Ref: Pérez-Salinas et al., Quantum 2020
# ================================================================

def enc_angle_ryrz(x_feat, weights):
    qv = QuantumVariable(N_Q)
    # ENCODING: two angles per qubit (feature pairs)
    for i in range(N_Q):
        ry(x_feat[i], qv[i])
        rz(x_feat[(i + N_Q // 2) % N_Q], qv[i])   # complementary feature
    # VARIATIONAL
    for l in range(L):
        variational_layer(qv, weights, N_Q, l)
    return measure_all(qv)


# ================================================================
# ENCODING 3: AMPLITUDE ENCODING
# ================================================================
# Feature vector → quantum state amplitudes.
# Exponential compression: n features in log₂(n) qubits.
# Ref: Mottonen et al. 2004; Grover & Rudolph 2002
#
# Qrisp note: use qs.init_state(statevector) for amplitude init.
# ================================================================

def enc_amplitude(x_feat, weights):
    qv = QuantumVariable(N_Q_AMP)
    # ENCODING: initialise as amplitude statevector
    x_padded = pad_amplitude(x_feat, N_Q_AMP)
    qv.qs.init_state(x_padded)          # Qrisp statevector initialisation
    # VARIATIONAL (RY+RZ + CZ ladder)
    for l in range(L):
        for i in range(N_Q_AMP):
            ry(weights[l, i, 0], qv[i])
            rz(weights[l, i, 1], qv[i])
        for i in range(N_Q_AMP - 1):
            cz(qv[i], qv[i + 1])
    return measure_all(qv)


# ================================================================
# ENCODING 4: BASIS ENCODING
# ================================================================
# Discretises features → binary → computational basis state.
# Pros: Trivial circuit. Cons: Loses continuous information.
# Ref: Nielsen & Chuang, 2010
# ================================================================

def enc_basis(x_int, weights):
    qv = QuantumVariable(N_Q_BASIS)
    # ENCODING: flip qubits corresponding to '1' bits
    idx  = int(np.clip(x_int, 0, 2 ** N_Q_BASIS - 1))
    b    = bits(idx, N_Q_BASIS)
    for i, bit in enumerate(b):
        if bit:
            x(qv[i])                    # |0⟩ → |1⟩
    # VARIATIONAL
    for l in range(L):
        for i in range(N_Q_BASIS):
            ry(weights[l, i, 0], qv[i])
        for i in range(N_Q_BASIS - 1):
            cz(qv[i], qv[i + 1])
    return measure_all(qv)


# ================================================================
# ENCODING 5: ZZ FEATURE MAP
# ================================================================
# Two-body interaction encoding — maps data to quantum kernel space.
# Ref: Havlíček et al., Nature 2019
# ================================================================

def enc_zz_feature_map(x_feat, weights):
    qv = QuantumVariable(N_Q)
    # FIRST LAYER: Hadamard + single-qubit phases
    for i in range(N_Q):
        h(qv[i])
    for i in range(N_Q):
        p(x_feat[i], qv[i])             # PhaseShift ≡ p gate in Qrisp
    # ZZ INTERACTIONS (nearest-neighbour)
    for i in range(N_Q - 1):
        cz(qv[i], qv[i + 1])
        p(x_feat[i] * x_feat[i + 1], qv[i])    # ZZ cross-phase
        cz(qv[i], qv[i + 1])
    # SECOND LAYER (depth-2 feature map)
    for i in range(N_Q):
        h(qv[i])
    for i in range(N_Q):
        p(x_feat[i], qv[i])
    # VARIATIONAL
    for l in range(L):
        variational_layer(qv, weights, N_Q, l)
    return measure_all(qv)


# ================================================================
# ENCODING 6: IQP (Instantaneous Quantum Polynomial)
# ================================================================
# Diagonal gates in Hadamard basis — classically hard to simulate.
# Ref: Shepherd & Bremner 2009; Havlíček et al. 2019
# ================================================================

def enc_iqp(x_feat, weights):
    qv = QuantumVariable(N_Q)
    # LAYER 1: Hadamard
    for i in range(N_Q):
        h(qv[i])
    # DIAGONAL GATES: RZ(xᵢ) + CZ interactions
    for i in range(N_Q):
        rz(x_feat[i], qv[i])
    for i in range(0, N_Q - 1, 2):
        cz(qv[i], qv[i + 1])
        rz(x_feat[i] * x_feat[i + 1], qv[i + 1])   # cross-term
    # LAYER 2: Hadamard
    for i in range(N_Q):
        h(qv[i])
    # VARIATIONAL
    for l in range(L):
        variational_layer(qv, weights, N_Q, l)
    return measure_all(qv)


# ================================================================
# ENCODING 7: HAMILTONIAN ENCODING (Trotterised)
# ================================================================
# H(x) = Σᵢ xᵢ Zᵢ + Σᵢⱼ xᵢxⱼ ZᵢZⱼ  →  |ψ⟩ = e^{-iH(x)t}|0⟩
# Ref: Lloyd et al. PRX Quantum 2020
# ================================================================

def enc_hamiltonian(x_feat, weights, t=1.0, trotter_steps=2):
    qv = QuantumVariable(N_Q)
    # INITIAL STATE: uniform superposition
    for i in range(N_Q):
        h(qv[i])
    # TROTTERISED EVOLUTION
    dt = t / trotter_steps
    for _ in range(trotter_steps):
        # Single-body: Zᵢ
        for i in range(N_Q):
            rz(2 * x_feat[i] * dt, qv[i])
        # Two-body: ZᵢZⱼ (nearest-neighbour)
        for i in range(N_Q - 1):
            cz(qv[i], qv[i + 1])
            rz(2 * x_feat[i] * x_feat[i + 1] * dt, qv[i + 1])
            cz(qv[i], qv[i + 1])
    # VARIATIONAL
    for l in range(L):
        variational_layer(qv, weights, N_Q, l)
    return measure_all(qv)


# ================================================================
# ENCODING 8: QAOA-INSPIRED ENCODING
# ================================================================
# Alternates cost (data-dependent) and mixer layers.
# Ref: Farhi, Goldstone & Gutmann 2014
# ================================================================

def enc_qaoa(x_feat, weights, gammas=None):
    if gammas is None:
        gammas = x_feat
    qv = QuantumVariable(N_Q)
    # INITIAL STATE: uniform superposition
    for i in range(N_Q):
        h(qv[i])
    # QAOA LAYERS: alternate cost + mixer
    for l in range(L):
        # Cost layer: data-dependent phase
        for i in range(N_Q):
            rz(2 * gammas[i % len(gammas)], qv[i])
        for i in range(N_Q - 1):
            cz(qv[i], qv[i + 1])
        # Mixer layer: X rotations
        beta = weights[l, 0, 0]
        for i in range(N_Q):
            rx(2 * beta, qv[i])
    return measure_all(qv)


# ================================================================
# ENCODING 9: SQUEEZING-INSPIRED (DV approximation)
# ================================================================
# Mimics CV squeezing: amplifies one quadrature, compresses the other.
# Approximated with RZ(x²) (quadratic phase space mapping).
# Ref: Killoran et al., Quantum 2019
# ================================================================

def enc_squeezing(x_feat, weights):
    qv = QuantumVariable(N_Q)
    # DISPLACEMENT LAYER (coherent amplitude)
    for i in range(N_Q):
        rx(x_feat[i], qv[i])
    # SQUEEZING LAYER (quadratic phase amplification)
    for i in range(N_Q):
        rz(x_feat[i] ** 2, qv[i])
    # BEAM-SPLITTER-LIKE (two-mode mixing via CZ + RY)
    for i in range(0, N_Q - 1, 2):
        cz(qv[i], qv[i + 1])
        ry( np.pi / 4, qv[i])
        ry(-np.pi / 4, qv[i + 1])
        cz(qv[i], qv[i + 1])
    # VARIATIONAL
    for l in range(L):
        variational_layer(qv, weights, N_Q, l)
    return measure_all(qv)


# ================================================================
# ENCODING 10: HYBRID (Angle + Amplitude)
# ================================================================
# First half → angle encoding (4 qubits)
# Second half → amplitude encoding (4 qubits)
# Ref: Derouich et al. 2025
# ================================================================

def enc_hybrid(x_feat, weights):
    qv    = QuantumVariable(N_Q)
    N_HALF = N_Q // 2                   # 4

    # ANGLE PART (qubits 0–3)
    for i in range(N_HALF):
        rx(x_feat[i], qv[i])

    # AMPLITUDE PART (qubits 4–7)
    x_amp = pad_amplitude(x_feat[N_HALF:], N_HALF)
    # Initialise amplitude sub-register via statevector on those qubits
    # (We embed into a fresh sub-variable then merge via entanglement)
    sub = QuantumVariable(N_HALF)
    sub.qs.init_state(x_amp)
    # Manually bind sub qubits into qv's session by entangling
    for i in range(N_HALF):
        cz(qv[N_HALF - 1], sub[i])     # bridge: links sessions

    # INTRA-BLOCK ENTANGLEMENT
    for i in range(N_HALF - 1):
        cz(qv[i], qv[i + 1])
    for i in range(N_HALF - 1):
        cz(sub[i], sub[i + 1])

    # VARIATIONAL on full 8-qubit register (qv carries the session)
    for l in range(L):
        variational_layer(qv, weights, N_Q, l)
    return measure_all(qv)


# ================================================================
# ENCODING 11: DISPLACEMENT-INSPIRED (CV-DV bridge)
# ================================================================
# Encodes complex features as (amplitude, phase) pairs via RX+RY.
# Ref: Derouich et al. 2025
# ================================================================

def enc_displacement(x_feat, weights):
    qv = QuantumVariable(N_Q)
    # DISPLACEMENT: (amplitude, phase) pairs
    for i in range(N_Q):
        alpha = x_feat[i]
        phi   = x_feat[(i + 1) % N_Q]
        rx(2 * alpha * np.cos(phi), qv[i])   # Re(α)
        ry(2 * alpha * np.sin(phi), qv[i])   # Im(α)
    # ENTANGLEMENT
    for i in range(N_Q - 1):
        cz(qv[i], qv[i + 1])
    # VARIATIONAL
    for l in range(L):
        variational_layer(qv, weights, N_Q, l)
    return measure_all(qv)


# ================================================================
# ENCODING 12: DATA RE-UPLOADING
# ================================================================
# Features injected at every variational layer → quantum Fourier series.
# Universal function approximator via data re-uploading.
# Ref: Pérez-Salinas et al., Quantum 2020
# ================================================================

def enc_reuploading(x_feat, weights):
    qv = QuantumVariable(N_Q)
    for l in range(L + 1):
        # DATA LAYER (re-upload at every depth)
        for i in range(N_Q):
            rx(x_feat[i], qv[i])
            rz(x_feat[(i + l) % N_Q], qv[i])   # shifted re-upload
        if l < L:
            # VARIATIONAL LAYER
            for i in range(N_Q):
                ry(weights[l, i, 0], qv[i])
            # ENTANGLEMENT (even + odd layers)
            for i in range(0, N_Q - 1, 2):
                cz(qv[i], qv[i + 1])
            for i in range(1, N_Q - 1, 2):
                cz(qv[i], qv[i + 1])
    return measure_all(qv)


# ── Registry ──────────────────────────────────────────────────────────────────

ENCODING_REGISTRY = [
    {"id":  1, "name": "Angle RX",          "fn": enc_angle_rx,       "n_q": N_Q,       "x_key": "angle"},
    {"id":  2, "name": "Angle RY+RZ",        "fn": enc_angle_ryrz,     "n_q": N_Q,       "x_key": "angle"},
    {"id":  3, "name": "Amplitude",          "fn": enc_amplitude,      "n_q": N_Q_AMP,   "x_key": "amplitude"},
    {"id":  4, "name": "Basis",              "fn": enc_basis,          "n_q": N_Q_BASIS, "x_key": "basis"},
    {"id":  5, "name": "ZZ Feature Map",     "fn": enc_zz_feature_map, "n_q": N_Q,       "x_key": "phase"},
    {"id":  6, "name": "IQP",                "fn": enc_iqp,            "n_q": N_Q,       "x_key": "angle"},
    {"id":  7, "name": "Hamiltonian",        "fn": enc_hamiltonian,    "n_q": N_Q,       "x_key": "phase"},
    {"id":  8, "name": "QAOA-inspired",      "fn": enc_qaoa,           "n_q": N_Q,       "x_key": "phase"},
    {"id":  9, "name": "Squeezing",          "fn": enc_squeezing,      "n_q": N_Q,       "x_key": "angle"},
    {"id": 10, "name": "Hybrid",             "fn": enc_hybrid,         "n_q": N_Q,       "x_key": "angle"},
    {"id": 11, "name": "Displacement",       "fn": enc_displacement,   "n_q": N_Q,       "x_key": "angle"},
    {"id": 12, "name": "Data Re-uploading",  "fn": enc_reuploading,    "n_q": N_Q,       "x_key": "fourier"},
]

print("✅  All 12 encoding circuits defined (Qrisp).")
for enc in ENCODING_REGISTRY:
    print(f"  #{enc['id']:2d}  {enc['name']:<22}  qubits={enc['n_q']}  x_prep={enc['x_key']}")
