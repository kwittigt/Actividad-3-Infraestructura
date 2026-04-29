import time
import os
import numpy as np
import pandas as pd
import dask.dataframe as dd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")



N_REPETICIONES = 3
N_FILAS = 2_000_000          # 2M de filas → suficiente para evidenciar overhead
PARTICIONES_DASK = [2, 8, 32]  # 3 configuraciones de particionamiento
DATA_PATH = "/tmp/datos_experimento_C.parquet"




def generar_datos(n=N_FILAS):
    np.random.seed(42)
    categorias = ["cat_A", "cat_B", "cat_C", "cat_D", "cat_E"]
    regiones   = ["Norte", "Sur", "Este", "Oeste", "Centro"]

    df = pd.DataFrame({
        "id":        np.arange(n),
        "categoria": np.random.choice(categorias, n),
        "region":    np.random.choice(regiones, n),
        "valor_1":   np.random.randn(n) * 100 + 500,
        "valor_2":   np.random.rand(n) * 1000,
        "valor_3":   np.random.exponential(scale=50, size=n),
        "cantidad":  np.random.randint(1, 200, n),
        "calidad":   np.random.uniform(0, 1, n),
        "flag":      np.random.choice([True, False], n, p=[0.7, 0.3]),
    })
    return df


def guardar_parquet(df):
    df.to_parquet(DATA_PATH, index=False, engine="pyarrow")
    size_mb = os.path.getsize(DATA_PATH) / 1e6
    print(f"[INFO] Datos guardados: {DATA_PATH} ({size_mb:.1f} MB, {len(df):,} filas)")




def pipeline_pandas():
    
    df = pd.read_parquet(DATA_PATH)

    # Limpieza: eliminar nulos y valores atípicos
    df = df.dropna()
    df = df[df["calidad"] >= 0.05]
    df = df[df["valor_1"] > 0]

    # Feature derivado
    df["ingreso"] = df["valor_1"] * df["cantidad"]

    # Filtrado de negocio
    df = df[df["flag"] == True]

    # GroupBy + Agregación
    agg = df.groupby(["categoria", "region"]).agg(
        total_ingreso=("ingreso",  "sum"),
        promedio_v2  =("valor_2",  "mean"),
        conteo       =("id",       "count"),
        max_calidad  =("calidad",  "max"),
    ).reset_index()

    # Tabla de referencia para merge
    ref = pd.DataFrame({
        "categoria": ["cat_A", "cat_B", "cat_C", "cat_D", "cat_E"],
        "factor":    [1.1,      0.9,     1.2,     1.0,     1.15]
    })
    resultado = agg.merge(ref, on="categoria", how="left")
    resultado["ingreso_ajustado"] = resultado["total_ingreso"] * resultado["factor"]

    return resultado




def pipeline_dask(n_particiones):
    """Pipeline distribuido (local) con Dask."""
    ddf = dd.read_parquet(DATA_PATH)
    ddf = ddf.repartition(npartitions=n_particiones)

    # Limpieza
    ddf = ddf.dropna()
    ddf = ddf[ddf["calidad"] >= 0.05]
    ddf = ddf[ddf["valor_1"] > 0]

    # Feature
    ddf["ingreso"] = ddf["valor_1"] * ddf["cantidad"]

    # Filtrado
    ddf = ddf[ddf["flag"] == True]

    # GroupBy + Agregación
    agg = ddf.groupby(["categoria", "region"]).agg(
        total_ingreso=("ingreso",  "sum"),
        promedio_v2  =("valor_2",  "mean"),
        conteo       =("id",       "count"),
        max_calidad  =("calidad",  "max"),
    ).reset_index().compute()

    # Merge (en Pandas tras compute)
    ref = pd.DataFrame({
        "categoria": ["cat_A", "cat_B", "cat_C", "cat_D", "cat_E"],
        "factor":    [1.1,      0.9,     1.2,     1.0,     1.15]
    })
    resultado = agg.merge(ref, on="categoria", how="left")
    resultado["ingreso_ajustado"] = resultado["total_ingreso"] * resultado["factor"]

    return resultado




def medir(fn, *args, n=N_REPETICIONES):
    tiempos = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn(*args)
        tiempos.append(time.perf_counter() - t0)
    return np.mean(tiempos), np.std(tiempos)


def ejecutar_experimento_C():
    print("=" * 60)
    print("EXPERIMENTO C: Procesamiento Distribuido con Dask")
    print(f"  N={N_FILAS:,}  |  Particiones: {PARTICIONES_DASK}  |  Reps: {N_REPETICIONES}")
    print("=" * 60)

    # Generar y guardar datos
    df = generar_datos()
    guardar_parquet(df)

    resultados = {}

    # Pandas
    m_pd, s_pd = medir(pipeline_pandas)
    resultados["pandas"] = (m_pd, s_pd)
    print(f"\n  Pandas monolítico : {m_pd:.4f}s ± {s_pd:.4f}s")

    # Dask con distintas particiones
    for npart in PARTICIONES_DASK:
        m_dk, s_dk = medir(pipeline_dask, npart)
        resultados[f"dask_{npart}"] = (m_dk, s_dk)
        ratio = m_pd / m_dk
        print(f"  Dask ({npart:2d} partic.) : {m_dk:.4f}s ± {s_dk:.4f}s  →  ratio={ratio:.2f}×")

    return resultados, m_pd


def graficar_C(resultados, m_pd):
    etiquetas = ["Pandas"] + [f"Dask\n({p} part.)" for p in PARTICIONES_DASK]
    claves    = ["pandas"] + [f"dask_{p}" for p in PARTICIONES_DASK]
    medias    = [resultados[k][0] for k in claves]
    stds      = [resultados[k][1] for k in claves]
    colores   = ["#e74c3c"] + ["#3498db", "#2ecc71", "#9b59b6"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Experimento C – Procesamiento Distribuido (Dask vs Pandas)",
                 fontsize=13, fontweight='bold')

    # Gráfico 1: Tiempos
    ax = axes[0]
    bars = ax.bar(etiquetas, medias, yerr=stds, color=colores, capsize=5, alpha=0.85)
    ax.set_ylabel("Tiempo (s)"); ax.set_title("Tiempo de ejecución del pipeline")
    ax.grid(alpha=0.3, axis='y')
    for bar, m in zip(bars, medias):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"{m:.2f}s", ha='center', va='bottom', fontsize=9)

    # Gráfico 2: Speedup relativo a Pandas
    ax2 = axes[1]
    speedups = [m_pd / m for m in medias]
    ax2.bar(etiquetas, speedups, color=colores, alpha=0.85)
    ax2.axhline(1.0, color='gray', linestyle='--', linewidth=1, label='Baseline (Pandas=1×)')
    ax2.set_ylabel("Speedup (×)"); ax2.set_title("Speedup relativo a Pandas")
    ax2.legend(); ax2.grid(alpha=0.3, axis='y')
    for i, s in enumerate(speedups):
        ax2.text(i, s + 0.02, f"{s:.2f}×", ha='center', va='bottom', fontsize=9)

    # Gráfico 3: Overhead relativo (cuánto más lento es Dask cuando lo es)
    ax3 = axes[2]
    overhead_pct = [(m / m_pd - 1) * 100 for m in medias[1:]]
    parts_labels = [f"{p} part." for p in PARTICIONES_DASK]
    colores_oh = ["#e74c3c" if o > 0 else "#2ecc71" for o in overhead_pct]
    bars3 = ax3.bar(parts_labels, overhead_pct, color=colores_oh, alpha=0.85)
    ax3.axhline(0, color='black', linewidth=0.8)
    ax3.set_ylabel("Overhead (%) vs Pandas")
    ax3.set_title("Overhead de coordinación Dask\n(negativo = ventaja Dask)")
    ax3.grid(alpha=0.3, axis='y')
    for bar, o in zip(bars3, overhead_pct):
        ax3.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + (1 if o >= 0 else -3),
                 f"{o:+.1f}%", ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig("/mnt/user-data/outputs/experimento_C.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("\n[✓] Gráfico guardado: experimento_C.png")


if __name__ == "__main__":
    resultados, m_pd = ejecutar_experimento_C()
    graficar_C(resultados, m_pd)
