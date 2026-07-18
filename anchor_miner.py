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

# Принудительно UTF-8 для вывода (обход cp1251)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
console = Console(force_terminal=True, highlight=False)

TEST_MODE = False

# --- Параллельное вычисление Cn (якорная фаза) ---
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

# Функция theta (формула Зигеля) — mod 2pi для безопасного cos() на GPU
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

    checkpoint_file = "checkpoint.json"
    delta_max = 50.0
    step = 0.001
    num_steps = int(math.ceil(delta_max / step))
    cores = cpu_count()

    # --- Восстановление или свежий старт ---
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
            console.print("[bold yellow][WARNING] Checkpoint corrupted. Starting fresh.[/bold yellow]")
    else:
        t_anchor_str  = "10000000000000.0"
        t_current_str = t_anchor_str
        iteration     = 1

    t_anchor  = mpmath.mpf(t_anchor_str)
    t_current = mpmath.mpf(t_current_str)
    N = int(mpmath.floor(mpmath.sqrt(t_anchor / (2 * pi))))

    # --- Генерация Cn.bin (только один раз) ---
    if os.path.exists(checkpoint_file) and os.path.exists("Cn.bin"):
        console.print(f"[green][INFO] Anchor Cn.bin found. N={N:,}. Skipping generation.[/green]")
    else:
        console.print(f"\n[INFO] ANCHOR GENERATION (one-time, using {cores} CPU cores)...")
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
        console.print(f"[green][INFO] Anchor ready in {time.time()-t0:.1f}s. Starting GPU miner![/green]")

    # --- Таблица вывода ---
    table = Table(show_header=True, header_style="bold magenta", border_style="bright_blue")
    table.add_column("Block",     justify="right",  style="cyan")
    table.add_column("Height t",  justify="right",  style="white")
    table.add_column("Expected",  justify="right",  style="yellow")
    table.add_column("GPU found", justify="right",  style="green")
    table.add_column("Time (s)",  justify="right",  style="white")
    table.add_column("Status",    justify="center")

    with Live(table, refresh_per_second=4, console=console):
        while True:
            t_block_start = time.time()

            # Python вычисляет theta mod 2pi с 40-значной точностью (быстро!)
            with open("theta.bin", "wb") as f:
                for k in range(num_steps):
                    t_k = t_current + mpmath.mpf(k) * mpmath.mpf(str(step))
                    th  = theta_mod2pi(t_k)
                    f.write(struct.pack("d", float(th)))

            expected_rounded = round(float(N_t(t_current + delta_max) - N_t(t_current)))

            # GPU получает N, t_anchor, t_current
            result = subprocess.run(
                ["riemann_anchor.exe",
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

            with open("miner_history.log", "a", encoding="utf-8") as f:
                f.write(f"| Block {iteration} | t={float(t_current):.1f}"
                        f" | Expected:{expected_rounded} | GPU:{gpu_zeros}"
                        f" | Time:{time_taken:.2f}s\n")

            if anomaly:
                with open("ANOMALY_FOUND.txt", "a", encoding="utf-8") as f:
                    f.write(f"=== ANOMALY ===\n"
                            f"t = {float(t_current):.6f}\n"
                            f"Expected: {expected_rounded} | GPU: {gpu_zeros}\n"
                            f"Diff: {abs(gpu_zeros - expected_rounded)}\n"
                            f"===============\n\n")
                console.print(Panel(
                    f"[bold red]!!! ANOMALY AT t = {float(t_current):.6f} !!![/bold red]\n"
                    f"Expected: {expected_rounded} | GPU found: {gpu_zeros}\n"
                    f"Saved to: ANOMALY_FOUND.txt",
                    title="!!! POTENTIAL RIEMANN HYPOTHESIS COUNTEREXAMPLE !!!",
                    border_style="red"
                ))
                for _ in range(5):
                    winsound.Beep(1000, 500)
                    time.sleep(0.3)
                    winsound.Beep(2000, 500)
                break

            # Атомарное сохранение — безопасно при выключении питания
            t_current += delta_max
            iteration += 1
            state = {"t_anchor": str(t_anchor),
                     "t_current": str(t_current),
                     "iteration": iteration}
            tmp = checkpoint_file + ".tmp"
            with open(tmp, "w") as f:
                json.dump(state, f)
            os.replace(tmp, checkpoint_file)

            if TEST_MODE and iteration > 5:
                console.print("[yellow]TEST MODE: Stopping.[/yellow]")
                break
