# Riemann Hypothesis GPU Miner

A high-performance GPU-accelerated scanner for zeros of the Riemann Zeta function, designed to search for potential counterexamples to the Riemann Hypothesis.

## What is this?

The **Riemann Hypothesis** is one of the most famous unsolved problems in mathematics, with a **$1,000,000 prize** offered by the Clay Mathematics Institute for its proof or disproof.

It states that all non-trivial zeros of the Riemann Zeta function lie on the "critical line" at real part = 1/2. This program continuously scans for zeros that might violate this rule.

## How it works

This miner uses the **Riemann-Siegel formula** combined with **Turing's Method** to:
1. Compute the exact expected number of zeros in a given interval using the von Mangoldt formula
2. Count the actual zeros found by the GPU on the critical line
3. Report any discrepancy (anomaly) that could indicate a counterexample

### Architecture

```
Python (mpmath, 40-digit precision)
  └─ Computes anchor phase Cn.bin ONE TIME
  └─ Computes theta(t) mod 2π for each block (safe for GPU)
       |
       v
CUDA C++ (RTX GPU)
  └─ Reads Cn.bin and theta.bin
  └─ Scans 50,000 points per block
  └─ Each point sums 1,261,566 terms: Σ cos(θ(t) + Cn - Δt·log(n)) / √n
  └─ Counts sign changes (zeros)
       |
       v
Python (orchestrator)
  └─ Saves checkpoint.json after every block
  └─ Logs results to miner_history.log
  └─ Alerts loudly if anomaly detected (ANOMALY_FOUND.txt + sound)
  └─ Auto-resumes from checkpoint on restart
```

## Requirements

- **GPU**: NVIDIA GPU with CUDA support (tested on RTX 3050)
- **CUDA Toolkit**: 11.0+
- **Visual Studio 2022** with C++ Desktop Development
- **Python 3.10+** with packages: `mpmath`, `rich`

Install Python dependencies:
```bash
pip install mpmath rich
```

## Quick Start

1. Clone the repository
2. Double-click `START_MINER.bat`
3. Wait ~6 seconds for the anchor to generate (one-time only)
4. Watch the GPU scan zeros at 10 trillion height!

The miner will **automatically resume** from where it left off if interrupted.

## Performance

| Metric | Value |
|--------|-------|
| Starting height | 10,000,000,000,000 (10 trillion) |
| Block size | 50 units of height |
| Steps per block | 50,000 |
| Terms per step | ~1,261,566 |
| Operations per block | ~63 billion |
| Time per block | ~30 seconds (RTX 3050) |
| Blocks per 8 hours | ~960 |
| Height covered in 8h | ~48,000 units |

## Files

| File | Purpose |
|------|---------|
| `START_MINER.bat` | **Main launcher** - double click to start |
| `riemann_anchor.cu` | CUDA C++ kernel (all heavy math) |
| `anchor_miner.py` | Python orchestrator (checkpoint, logging, display) |
| `Cn.bin` | Anchor phases (auto-generated, 10MB) |
| `miner_history.log` | Full log of all scanned blocks |
| `checkpoint.json` | Progress save file (auto-resume on restart) |
| `ANOMALY_FOUND.txt` | Created ONLY if a counterexample is found |

## If an anomaly is found

If the miner detects a discrepancy > 3 zeros:
1. The console shows a **large red alert panel**
2. The PC **beeps 5 times** as an alarm
3. All details are saved to `ANOMALY_FOUND.txt`
4. The miner **freezes** so you can document the finding

## Mathematical Background

The Riemann-Siegel formula approximates Z(t) as:

$$Z(t) = 2 \sum_{n=1}^{N} \frac{\cos(\theta(t) - t\ln n)}{\sqrt{n}} + R(t)$$

where $\theta(t) = \frac{t}{2}\ln\frac{t}{2\pi} - \frac{t}{2} - \frac{\pi}{8} + \frac{1}{48t}$ and $N = \lfloor\sqrt{t/2\pi}\rfloor$.

The expected zero count is given by the **von Mangoldt formula**:

$$N(T) = \frac{T}{2\pi}\ln\frac{T}{2\pi} - \frac{T}{2\pi} + \frac{7}{8} + O(\ln T)$$

## Why 10 trillion?

- Zeros up to ~10^22 have been verified on supercomputers
- We start at 10^13 as a validated baseline
- The unverified "dark matter" zone lies ahead — where anomalies have the best chance of hiding

## License

MIT License — do whatever you want with this. If you find a counterexample, please cite the project 😄
