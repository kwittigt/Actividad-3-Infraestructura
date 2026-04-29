import time
import math
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import psutil
import os
import warnings
warnings.filterwarnings("ignore")


N_REPETICIONES = 3
TAMANIOS = [100_000, 500_000, 1_000_000, 5_000_000, 10_000_000]
N_WORKERS = mp.cpu_count()

print(f"[INFO] Cores disponibles: {N_WORKERS}")
print(f"[INFO] RAM total: {psutil.virtual_memory().total / 1e9:.1f} GB")
print(f"[INFO] PID actual: {os.getpid()}\n")




def transformacion_secuencial(arr):
    """
    Transformación intensiva en Python puro:
    Para cada elemento x: sqrt(|x|) + x^2 / (|x| + 1e-9) + cumsum simulado
    """
    resultado = []
    acum = 0.0
    for x in arr:
        val = math.sqrt(abs(x)) + (x ** 2) / (abs(x) + 1e-9)
        acum += val
        resultado.append(acum)
    return resultado


def transformacion_numpy(arr):
    """
    Misma transformación usando NumPy vectorizado.
    """
    abs_arr = np.abs(arr)
    val = np.sqrt(abs_arr) + (arr ** 2) / (abs_arr + 1e-9)
    return np.cumsum(val)


def _procesar_bloque(args):
    """Función auxiliar para multiprocessing (debe ser top-level)."""
    bloque, offset = args
    abs_b = np.abs(bloque)
    val = np.sqrt(abs_b) + (bloque ** 2) / (abs_b + 1e-9)
    return np.cumsum(val)


def transformacion_paralela(arr, n_workers):
    """
    Divide el arreglo en bloques y procesa en paralelo con multiprocessing.
    """
    bloques = np.array_split(arr, n_workers)
    args = [(b, i) for i, b in enumerate(bloques)]
    with mp.Pool(processes=n_workers) as pool:
        resultados = pool.map(_procesar_bloque, args)
    # Reajustar cumsum: sumar el último valor acumulado del bloque anterior
    salida = [resultados[0]]
    for i in range(1, len(resultados)):
        offset = salida[i - 1][-1]
        salida.append(resultados[i] + offset)
    return np.concatenate(salida)




def medir_tiempo(fn, *args, n=N_REPETICIONES):
    tiempos = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn(*args)
        tiempos.append(time.perf_counter() - t0)
    return np.mean(tiempos), np.std(tiempos)


def ejecutar_experimento_A():
    resultados = {
        "tamanios": TAMANIOS,
        "sec_mean": [], "sec_std": [],
        "np_mean":  [], "np_std":  [],
        "par_mean": [], "par_std": [],
    }

    print("=" * 60)
    print("EXPERIMENTO A: Paralelismo de Datos y Vectorización")
    print("=" * 60)

    for n in TAMANIOS:
        arr = np.random.randn(n).astype(np.float64)
        arr_list = arr.tolist()

        print(f"\n[N={n:,}]")

        if n <= 1_000_000:
            m, s = medir_tiempo(transformacion_secuencial, arr_list)
            print(f"  Secuencial : {m:.4f}s ± {s:.4f}s")
        else:
            # Para arreglos muy grandes, Python puro es demasiado lento
            m, s = None, None
            print(f"  Secuencial : OMITIDO (N>{1_000_000:,} → tiempo excesivo)")

        resultados["sec_mean"].append(m)
        resultados["sec_std"].append(s)

        m_np, s_np = medir_tiempo(transformacion_numpy, arr)
        print(f"  NumPy      : {m_np:.4f}s ± {s_np:.4f}s")
        resultados["np_mean"].append(m_np)
        resultados["np_std"].append(s_np)

        m_p, s_p = medir_tiempo(transformacion_paralela, arr, N_WORKERS)
        print(f"  Paralelo   : {m_p:.4f}s ± {s_p:.4f}s ({N_WORKERS} workers)")
        resultados["par_mean"].append(m_p)
        resultados["par_std"].append(s_p)

    return resultados


def graficar_A(res):
    n_labels = [f"{n//1000}K" if n < 1_000_000 else f"{n//1_000_000}M"
                for n in res["tamanios"]]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Experimento A – Vectorización y Paralelismo de Datos", fontsize=14, fontweight='bold')

    # Gráfico 1: Tiempos de ejecución
    ax = axes[0]
    x = np.arange(len(n_labels))
    idx_sec = [i for i, v in enumerate(res["sec_mean"]) if v is not None]

    if idx_sec:
        ax.errorbar([x[i] for i in idx_sec],
                    [res["sec_mean"][i] for i in idx_sec],
                    [res["sec_std"][i] for i in idx_sec],
                    marker='o', label='Python puro', color='#e74c3c', capsize=4)

    ax.errorbar(x, res["np_mean"], res["np_std"],
                marker='s', label='NumPy', color='#2ecc71', capsize=4)
    ax.errorbar(x, res["par_mean"], res["par_std"],
                marker='^', label=f'Multiprocessing ({N_WORKERS}w)', color='#3498db', capsize=4)
    ax.set_xticks(x); ax.set_xticklabels(n_labels)
    ax.set_xlabel("Tamaño de entrada (N)"); ax.set_ylabel("Tiempo (s)")
    ax.set_title("Tiempo de ejecución"); ax.legend(); ax.grid(alpha=0.3)

    # Gráfico 2: Speedup de NumPy vs secuencial (donde hay datos)
    ax2 = axes[1]
    speedup_np  = [res["sec_mean"][i] / res["np_mean"][i]  for i in idx_sec]
    speedup_par = [res["sec_mean"][i] / res["par_mean"][i] for i in idx_sec]
    labels_sec = [n_labels[i] for i in idx_sec]
    xx = np.arange(len(idx_sec))
    ax2.bar(xx - 0.2, speedup_np,  0.35, label='NumPy', color='#2ecc71')
    ax2.bar(xx + 0.2, speedup_par, 0.35, label=f'Paralelo ({N_WORKERS}w)', color='#3498db')
    ax2.axhline(1, color='gray', linestyle='--', linewidth=0.8)
    ax2.set_xticks(xx); ax2.set_xticklabels(labels_sec)
    ax2.set_xlabel("Tamaño de entrada"); ax2.set_ylabel("Speedup (×)")
    ax2.set_title("Speedup vs Python puro"); ax2.legend(); ax2.grid(alpha=0.3, axis='y')

    # Gráfico 3: Eficiencia del paralelo
    ax3 = axes[2]
    efic = [s / N_WORKERS for s in speedup_par]
    ax3.plot(labels_sec, efic, marker='D', color='#9b59b6', linewidth=2)
    ax3.axhline(1.0, color='gray', linestyle='--', linewidth=0.8, label='Ideal (100%)')
    ax3.set_ylim(0, 1.2)
    ax3.set_xlabel("Tamaño de entrada"); ax3.set_ylabel("Eficiencia (Speedup / workers)")
    ax3.set_title(f"Eficiencia del paralelismo ({N_WORKERS} workers)"); ax3.legend(); ax3.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("/mnt/user-data/outputs/experimento_A.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("\n[✓] Gráfico guardado: experimento_A.png")


if __name__ == "__main__":
    res = ejecutar_experimento_A()
    graficar_A(res)
