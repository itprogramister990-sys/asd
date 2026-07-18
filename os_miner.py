import sys
import io
import mpmath
import math
import struct
import subprocess
import time
import os
import json
import winsound
from multiprocessing import Pool, cpu_count
from rich.live import Live
from rich.table import Table
from rich.console import Console
from rich.panel import Panel

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
console = Console(force_terminal=True, highlight=False)

TEST_MODE = False  # Тест пройден! FP32 даёт те же результаты что FP64.

def compute_chunk(args):
    start_n, end_n, t_anchor_str = args
    mpmath.mp.dps = 40
    t_anch = mpmath.mpf(t_anchor_str)
    pi = mpmath.pi
    chunk_data = bytearray()
    for n in range(start_n, end_n + 1):
        val = -t_anch * mpmath.log(n)
        val_mod = mpmath.fmod(val, 2 * pi)
        if val_mod < 0:
            val_mod += 2 * pi
        chunk_data += struct.pack("d", float(val_mod))  # FP64 Cn.bin (ради совместимости)
    return chunk_data

def theta_mod2pi(t):
    pi = mpmath.pi
    raw = (t / 2) * mpmath.log(t / (2 * pi)) - (t / 2) - pi / 8 + 1 / (48 * t)
    mod = mpmath.fmod(raw, 2 * pi)
    if mod < 0:
        mod += 2 * pi
    return mod

def N_t(t):
    pi = mpmath.pi
    return (t / (2 * pi)) * mpmath.log(t / (2 * pi)) - (t / (2 * pi)) + mpmath.mpf("7") / 8

if __name__ == '__main__':
    mpmath.mp.dps = 40
    pi = mpmath.pi

    checkpoint_file = "checkpoint_os.json"
    delta_max = 50.0
    step = 0.001
    num_steps = int(math.ceil(delta_max / step))
    cores = cpu_count()

    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, "r") as f:
                state = json.load(f)
            t_anchor_str  = state["t_anchor"]
            t_current_str = state["t_current"]
            iteration     = state["iteration"]
            console.print(f"[bold cyan][RESUME][/bold cyan] Block {iteration}, t = {t_current_str}")
        except Exception:
            t_anchor_str  = "10000000000000.0"
            t_current_str = t_anchor_str
            iteration     = 1
    else:
        t_anchor_str  = "10000000000000.0"
        t_current_str = t_anchor_str
        iteration     = 1

    t_anchor  = mpmath.mpf(t_anchor_str)
    t_current = mpmath.mpf(t_current_str)
    N = int(mpmath.floor(mpmath.sqrt(t_anchor / (2 * pi))))

    # Cn.bin: используем тот же что у anchor_miner.py (FP64, совместимо)
    if os.path.exists(checkpoint_file) and os.path.exists("Cn.bin"):
        console.print(f"[green][INFO] Anchor Cn.bin found. N={N:,}. Skipping generation.[/green]")
    else:
        console.print(f"\n[INFO] ANCHOR GENERATION ({cores} cores)...")
        t0 = time.time()
        chunk_size = math.ceil(N / cores)
        ranges = []
        for i in range(cores):
            sn = i * chunk_size + 1
            en = min((i + 1) * chunk_size, N)
            if sn <= N:
                ranges.append((sn, en, t_anchor_str))
        with Pool(processes=cores) as pool:
            results = pool.map(compute_chunk, ranges)
        with open("Cn.bin", "wb") as f:
            for res in results:
                f.write(res)
        console.print(f"[green][INFO] Anchor done in {time.time()-t0:.1f}s.[/green]")

    console.print(f"[bold yellow][OS-MINER][/bold yellow] FP32 mode | N={N:,} | Target speedup: ~32x vs FP64")

    table = Table(show_header=True, header_style="bold cyan", border_style="bright_blue")
    table.add_column("Block",     justify="right",  style="cyan")
    table.add_column("Height t",  justify="right",  style="white")
    table.add_column("Expected",  justify="right",  style="yellow")
    table.add_column("GPU FP32",  justify="right",  style="green")
    table.add_column("Time (s)",  justify="right",  style="white")
    table.add_column("Status",    justify="center")

    with Live(table, refresh_per_second=4, console=console):
        while True:
            t_block_start = time.time()

            # Theta в FP32 (4 байта вместо 8 = меньше трафик)
            with open("theta_f32.bin", "wb") as f:
                for k in range(num_steps):
                    t_k = t_current + mpmath.mpf(k) * mpmath.mpf(str(step))
                    th  = theta_mod2pi(t_k)
                    f.write(struct.pack("f", float(th)))  # <- FP32!

            expected_rounded = round(float(N_t(t_current + delta_max) - N_t(t_current)))

            result = subprocess.run(
                ["riemann_os.exe",
                 str(N), str(float(t_anchor)), str(float(t_current))],
                capture_output=True, text=True
            )
            gpu_zeros = -1
            for line in result.stdout.split('\n'):
                if "[RESULT] ZEROS=" in line:
                    try:
                        gpu_zeros = int(line.split("=")[1].strip())
                    except Exception:
                        pass

            time_taken = time.time() - t_block_start
            anomaly = (abs(gpu_zeros - expected_rounded) > 3) or (gpu_zeros == -1)
            status  = "[bold red]!!! ANOMALY !!![/bold red]" if anomaly else "[bold green]OK[/bold green]"

            table.add_row(str(iteration), f"{float(t_current):.1f}",
                          str(expected_rounded), str(gpu_zeros),
                          f"{time_taken:.2f}", status)

            with open("miner_os.log", "a", encoding="utf-8") as f:
                f.write(f"| Block {iteration} | t={float(t_current):.1f}"
                        f" | Expected:{expected_rounded} | GPU_FP32:{gpu_zeros}"
                        f" | Time:{time_taken:.2f}s\n")

            if anomaly:
                # Проверяем: это баг FP32 или реальная аномалия?
                with open("ANOMALY_FP32.txt", "a", encoding="utf-8") as f:
                    f.write(f"t={float(t_current):.6f} | "
                            f"Expected:{expected_rounded} | FP32:{gpu_zeros} | "
                            f"VERIFY WITH FP64!\n")
                console.print("[bold yellow][WARNING] Anomaly detected — verify with anchor_miner.py before celebrating![/bold yellow]")
                for _ in range(3):
                    winsound.Beep(1500, 300)
                break

            t_current += delta_max
            iteration += 1
            state = {"t_anchor": str(t_anchor), "t_current": str(t_current),
                     "iteration": iteration}
            tmp = checkpoint_file + ".tmp"
            with open(tmp, "w") as f:
                json.dump(state, f)
            os.replace(tmp, checkpoint_file)

            if TEST_MODE and iteration > 5:
                console.print("[yellow]TEST done. Check results vs FP64. If OK — set TEST_MODE=False.[/yellow]")
                break
