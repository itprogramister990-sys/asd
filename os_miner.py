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
        chunk_data += struct.pack("d", float(val_mod))
    return chunk_data

def theta_raw(t):
    pi = mpmath.pi
    return (t / 2) * mpmath.log(t / (2 * pi)) - (t / 2) - pi / 8 + 1 / (48 * t)

def theta_mod2pi(t):
    pi = mpmath.pi
    mod = mpmath.fmod(theta_raw(t), 2 * pi)
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

    def generate_cn_bin(N_val, t_anch_str, cores_count):
        console.print(f"\n[bold cyan][ANCHOR][/bold cyan] Generating FP32 Cn.bin for t={t_anch_str} ({cores_count} cores)...")
        t0 = time.time()
        chunk_size = math.ceil(N_val / cores_count)
        ranges = []
        for i in range(cores_count):
            sn = i * chunk_size + 1
            en = min((i + 1) * chunk_size, N_val)
            if sn <= N_val:
                ranges.append((sn, en, t_anch_str))
        with Pool(processes=cores_count) as pool:
            results = pool.map(compute_chunk, ranges)
        with open("Cn.bin", "wb") as f:
            for res in results:
                f.write(res)
        console.print(f"[green][INFO] Anchor done in {time.time()-t0:.1f}s.[/green]")

    if os.path.exists("Cn.bin"):
        if os.path.getsize("Cn.bin") == N * 8:
            console.print(f"[green][INFO] Anchor Cn.bin found (FP64). N={N:,}. Skipping generation.[/green]")
        else:
            console.print(f"[ANCHOR] Generating FP64 Cn.bin for N={N:,} at anchor t={t_anchor_str}...")
            generate_cn_bin(N, t_anchor_str, cores)
    else:
        console.print(f"[ANCHOR] Generating FP64 Cn.bin for N={N:,} at anchor t={t_anchor_str}...")
        generate_cn_bin(N, t_anchor_str, cores)

    console.print(f"[bold yellow][OS-MINER][/bold yellow] Rigorous FP64 Interval Arithmetic mode | N={N:,} | AMR Enabled")

    table = Table(show_header=True, header_style="bold cyan", border_style="bright_blue")
    table.add_column("Block",     justify="right",  style="cyan")
    table.add_column("Height t",  justify="right",  style="white")
    table.add_column("Expected",  justify="right",  style="yellow")
    table.add_column("GPU Zeros", justify="right",  style="green")
    table.add_column("Warnings",  justify="right",  style="red")
    table.add_column("Time (s)",  justify="right",  style="white")
    table.add_column("Status",    justify="center")

    with Live(table, refresh_per_second=4, console=console):
        while True:
            t_block_start = time.time()

            # Anchor reset: limit delta_t to ensure cosf() remains highly accurate
            if (t_current - t_anchor) > 3900:
                t_anchor = t_current
                t_anchor_str = str(t_anchor)
                console.print(f"\n[bold magenta][ANCHOR RESET][/bold magenta] Delta_t > 3900. Regenerating anchor to maintain FP32 precision...")
                generate_cn_bin(N, t_anchor_str, cores)

            th_base_raw = theta_raw(t_current)
            th_next_raw = theta_raw(t_current + mpmath.mpf(str(step)))
            step_theta = float(th_next_raw - th_base_raw)
            theta_base = float(mpmath.fmod(th_base_raw, 2 * mpmath.pi))
            if theta_base < 0.0: theta_base += 2.0 * math.pi

            expected_rounded = round(float(N_t(t_current + delta_max) - N_t(t_current)))

            result = subprocess.run(
                ["riemann_os.exe",
                 str(N), str(float(t_anchor)), str(float(t_current)),
                 f"{theta_base:.15f}", f"{step_theta:.15f}"],
                capture_output=True, text=True
            )
            gpu_zeros = -1
            gpu_warnings = 0
            for line in result.stdout.split('\n'):
                if "[RESULT] ZEROS=" in line:
                    try:
                        parts = line.split("WARNINGS=")
                        gpu_warnings = int(parts[1].strip())
                        gpu_zeros = int(parts[0].split("ZEROS=")[1].strip())
                    except Exception:
                        pass

            time_taken = time.time() - t_block_start
            anomaly = (abs(gpu_zeros - expected_rounded) > 3) or (gpu_zeros == -1)
            
            if gpu_warnings > 0:
                status = f"[bold yellow]WARN ({gpu_warnings})[/bold yellow]"
            else:
                status = "[bold red]!!! ANOMALY !!![/bold red]" if anomaly else "[bold green]RIGOROUS[/bold green]"

            table.add_row(str(iteration), f"{float(t_current):.1f}",
                          str(expected_rounded), str(gpu_zeros), str(gpu_warnings),
                          f"{time_taken:.2f}", status)

            with open("miner_os.log", "a", encoding="utf-8") as f:
                f.write(f"| Block {iteration} | t={float(t_current):.1f}"
                        f" | Expected:{expected_rounded} | GPU_Zeros:{gpu_zeros}"
                        f" | Warnings:{gpu_warnings} | Time:{time_taken:.2f}s\n")

            if gpu_warnings > 0:
                with open("PRECISION_WARNINGS.txt", "a", encoding="utf-8") as f:
                    f.write(f"t={float(t_current):.6f} | Warnings: {gpu_warnings} | Interval Exhaustion Detected!\n")

            if anomaly:
                with open("ANOMALY_FP64.txt", "a", encoding="utf-8") as f:
                    f.write(f"t={float(t_current):.6f} | "
                            f"Expected:{expected_rounded} | Zeros:{gpu_zeros} | "
                            f"MATHEMATICAL ANOMALY DETECTED!\n")
                console.print("[bold yellow][WARNING] Anomaly detected in strict rigorous mode![/bold yellow]")
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
