import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import warnings
warnings.filterwarnings("ignore")



N_REPETICIONES = 3
TAMANIOS_LOTE = [500, 2_000, 10_000]          # filas por lote
N_LOTES = 12                                   # cantidad de lotes totales
CONFIGURACIONES_WORKERS = [2, 4, 8]




def feature_engineering(df_chunk):
    
    df = df_chunk.copy()

    # Features de razón
    df["ratio_A_B"]   = df["col_A"] / (df["col_B"] + 1e-9)
    df["ratio_C_D"]   = df["col_C"] / (df["col_D"] + 1e-9)

    # Interacciones polinómicas
    df["prod_AB"]     = df["col_A"] * df["col_B"]
    df["sq_A"]        = df["col_A"] ** 2
    df["sq_C"]        = df["col_C"] ** 2

    # Estadísticas por fila (simulando features de ventana)
    df["row_mean"]    = df[["col_A", "col_B", "col_C", "col_D"]].mean(axis=1)
    df["row_std"]     = df[["col_A", "col_B", "col_C", "col_D"]].std(axis=1)
    df["row_max"]     = df[["col_A", "col_B", "col_C", "col_D"]].max(axis=1)

    # Normalización manual (z-score por columna)
    for col in ["col_A", "col_B", "col_C", "col_D"]:
        mu = df[col].mean()
        sigma = df[col].std() + 1e-9
        df[f"{col}_norm"] = (df[col] - mu) / sigma

    # Agrupación ficticia por categoría
    df["categoria"] = pd.cut(df["col_A"], bins=5, labels=False)
    agg = df.groupby("categoria")["row_mean"].transform("mean")
    df["group_mean_A"] = agg

    return df


def generar_lotes(n_filas_por_lote, n_lotes):
    lotes = []
    for _ in range(n_lotes):
        df = pd.DataFrame({
            "col_A": np.random.randn(n_filas_por_lote),
            "col_B": np.random.rand(n_filas_por_lote) * 10,
            "col_C": np.random.randn(n_filas_por_lote) * 5,
            "col_D": np.random.rand(n_filas_por_lote),
        })
        lotes.append(df)
    return lotes



def ejecutar_secuencial(lotes):
    return [feature_engineering(lote) for lote in lotes]


def ejecutar_thread_pool(lotes, n_workers):
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        return list(ex.map(feature_engineering, lotes))


def ejecutar_process_pool(lotes, n_workers):
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        return list(ex.map(feature_engineering, lotes))


def medir(fn, *args, n=N_REPETICIONES):
    tiempos = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn(*args)
        tiempos.append(time.perf_counter() - t0)
    return np.mean(tiempos), np.std(tiempos)




def ejecutar_experimento_B():
    todos = {}

    print("=" * 60)
    print("EXPERIMENTO B: Paralelismo a nivel de Tareas (Feature Eng.)")
    print(f"  Lotes: {N_LOTES}  |  Repeticiones: {N_REPETICIONES}")
    print("=" * 60)

    for tam in TAMANIOS_LOTE:
        lotes = generar_lotes(tam, N_LOTES)
        clave = f"{tam}_filas"
        todos[clave] = {"workers": [], "secuencial": None,
                        "thread": {}, "process": {}}

        print(f"\n[Lote={tam:,} filas × {N_LOTES} lotes]")

        # Secuencial
        m_s, s_s = medir(ejecutar_secuencial, lotes)
        todos[clave]["secuencial"] = (m_s, s_s)
        print(f"  Secuencial : {m_s:.4f}s ± {s_s:.4f}s")

        for nw in CONFIGURACIONES_WORKERS:
            # Thread pool
            m_t, s_t = medir(ejecutar_thread_pool, lotes, nw)
            todos[clave]["thread"][nw] = (m_t, s_t)
            sp_t = m_s / m_t
            print(f"  ThreadPool  w={nw}: {m_t:.4f}s ± {s_t:.4f}s  →  speedup={sp_t:.2f}×")

            # Process pool
            m_p, s_p = medir(ejecutar_process_pool, lotes, nw)
            todos[clave]["process"][nw] = (m_p, s_p)
            sp_p = m_s / m_p
            print(f"  ProcessPool w={nw}: {m_p:.4f}s ± {s_p:.4f}s  →  speedup={sp_p:.2f}×")

    return todos



def graficar_B(todos):
    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    fig.suptitle("Experimento B – Paralelismo a Nivel de Tareas (Feature Engineering)",
                 fontsize=13, fontweight='bold')

    for col_idx, (clave, datos) in enumerate(todos.items()):
        tam_label = clave.replace("_filas", "").replace("_", ",")

        m_s = datos["secuencial"][0]
        workers = CONFIGURACIONES_WORKERS

        thread_means  = [datos["thread"][w][0]  for w in workers]
        process_means = [datos["process"][w][0] for w in workers]
        thread_stds   = [datos["thread"][w][1]  for w in workers]
        process_stds  = [datos["process"][w][1] for w in workers]

        # Fila 0: Tiempos absolutos
        ax = axes[0][col_idx]
        x = np.arange(len(workers))
        ax.bar(x - 0.2, thread_means,  0.35, yerr=thread_stds,
               label='ThreadPool', color='#3498db', capsize=4)
        ax.bar(x + 0.2, process_means, 0.35, yerr=process_stds,
               label='ProcessPool', color='#e67e22', capsize=4)
        ax.axhline(m_s, color='#e74c3c', linestyle='--', linewidth=1.5, label=f'Secuencial ({m_s:.2f}s)')
        ax.set_xticks(x); ax.set_xticklabels([f"{w}w" for w in workers])
        ax.set_title(f"Tiempos – lote {tam_label} filas"); ax.set_ylabel("Tiempo (s)")
        ax.legend(fontsize=7); ax.grid(alpha=0.3, axis='y')

        # Fila 1: Speedup
        ax2 = axes[1][col_idx]
        sp_t = [m_s / t for t in thread_means]
        sp_p = [m_s / t for t in process_means]
        ax2.plot(workers, sp_t, marker='o', label='ThreadPool',  color='#3498db', linewidth=2)
        ax2.plot(workers, sp_p, marker='s', label='ProcessPool', color='#e67e22', linewidth=2)
        ax2.plot(workers, workers, linestyle='--', color='gray', linewidth=0.8, label='Ideal lineal')
        ax2.set_xlabel("Nº de workers"); ax2.set_ylabel("Speedup (×)")
        ax2.set_title(f"Speedup – lote {tam_label} filas"); ax2.legend(fontsize=7); ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("/mnt/user-data/outputs/experimento_B.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("\n[✓] Gráfico guardado: experimento_B.png")


if __name__ == "__main__":
    todos = ejecutar_experimento_B()
    graficar_B(todos)
