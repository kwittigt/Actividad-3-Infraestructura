import time
import numpy as np
import pandas as pd
import multiprocessing as mp
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")



N_REPETICIONES = 3
WORKERS_RANGE   = [1, 2, 4, 6, 8, 10, 12, 16]
TAMANIO_DATOS   = 1_000_000   # elementos por experimento




def tarea_cpu_pura(args):
    """Tarea CPU-bound: cálculo intensivo sin overhead de comunicación."""
    bloque, _ = args
    arr = np.array(bloque, dtype=np.float64)
    resultado = np.sum(np.sqrt(np.abs(arr)) + arr ** 2)
    return resultado


def tarea_con_overhead(args):
   
    bloque, overhead_factor = args
    # Simular serialización/deserialización
    arr = np.array(bloque, dtype=np.float64)
    time.sleep(overhead_factor * len(arr) * 1e-8)   # overhead de coordinación
    resultado = np.sum(np.sqrt(np.abs(arr)) + arr ** 2)
    time.sleep(overhead_factor * len(arr) * 5e-9)   # overhead de resultado
    return resultado




def ejecutar_pipeline(datos, n_workers, overhead_factor=0.0):
    bloques = np.array_split(datos, n_workers)
    args = [(b.tolist(), overhead_factor) for b in bloques]

    if overhead_factor == 0.0:
        fn = tarea_cpu_pura
    else:
        fn = tarea_con_overhead

    with mp.Pool(processes=n_workers) as pool:
        resultados = pool.map(fn, args)
    return sum(resultados)


def tiempo_secuencial_base(datos):
    
    arr = np.asarray(datos, dtype=np.float64)
    return np.sum(np.sqrt(np.abs(arr)) + arr ** 2)


def medir(fn, *args, n=N_REPETICIONES):
    tiempos = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn(*args)
        tiempos.append(time.perf_counter() - t0)
    return np.mean(tiempos), np.std(tiempos)




def speedup_amdahl(p_paralelo, n_workers):
    """S(n) = 1 / ((1 - p) + p/n)"""
    return [1 / ((1 - p_paralelo) + p_paralelo / n) for n in n_workers]




def ejecutar_experimento_D():
    print("=" * 60)
    print("EXPERIMENTO D: Overhead de Coordinación y Decisión Técnica")
    print(f"  Datos: {TAMANIO_DATOS:,}  |  Workers: {WORKERS_RANGE}")
    print("=" * 60)

    datos = np.random.randn(TAMANIO_DATOS)

    # Tiempo secuencial de referencia
    m_sec, s_sec = medir(tiempo_secuencial_base, datos)
    print(f"\n  Secuencial (referencia): {m_sec:.4f}s ± {s_sec:.4f}s")

    resultados = {
        "sin_overhead": {"workers": [], "speedup": [], "eficiencia": [], "tiempo": []},
        "con_overhead": {"workers": [], "speedup": [], "eficiencia": [], "tiempo": []},
    }

    print(f"\n{'Workers':>8}  {'Sin OH (t)':>12}  {'Speedup':>8}  {'Con OH (t)':>12}  {'Speedup':>8}")
    print("-" * 60)

    for nw in WORKERS_RANGE:
        # Sin overhead adicional
        m_s, _ = medir(ejecutar_pipeline, datos, nw, 0.0)
        sp_s = m_sec / m_s
        ef_s = sp_s / nw
        resultados["sin_overhead"]["workers"].append(nw)
        resultados["sin_overhead"]["speedup"].append(sp_s)
        resultados["sin_overhead"]["eficiencia"].append(ef_s)
        resultados["sin_overhead"]["tiempo"].append(m_s)

        # Con overhead de coordinación
        m_o, _ = medir(ejecutar_pipeline, datos, nw, 1.5)
        sp_o = m_sec / m_o
        ef_o = sp_o / nw
        resultados["con_overhead"]["workers"].append(nw)
        resultados["con_overhead"]["speedup"].append(sp_o)
        resultados["con_overhead"]["eficiencia"].append(ef_o)
        resultados["con_overhead"]["tiempo"].append(m_o)

        print(f"  {nw:6d}  {m_s:12.4f}  {sp_s:8.2f}×  {m_o:12.4f}  {sp_o:8.2f}×")

    return resultados, m_sec, datos




def graficar_D(resultados, m_sec):
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle("Experimento D – Overhead de Coordinación y Límites del Speedup",
                 fontsize=14, fontweight='bold')

    gs = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.35)

    workers = WORKERS_RANGE
    ideal_speedup = workers

    sp_sin = resultados["sin_overhead"]["speedup"]
    sp_con = resultados["con_overhead"]["speedup"]
    ef_sin = resultados["sin_overhead"]["eficiencia"]
    ef_con = resultados["con_overhead"]["eficiencia"]
    t_sin  = resultados["sin_overhead"]["tiempo"]
    t_con  = resultados["con_overhead"]["tiempo"]

    # Proyecciones Amdahl para distintas fracciones paralelas
    amdahl_90 = speedup_amdahl(0.90, workers)
    amdahl_95 = speedup_amdahl(0.95, workers)
    amdahl_99 = speedup_amdahl(0.99, workers)

    # ── Gráfico 1: Speedup real vs Ley de Amdahl ──
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(workers, ideal_speedup, '--', color='gray', label='Ideal lineal', linewidth=1.5)
    ax1.plot(workers, amdahl_90, ':', color='#f39c12', label='Amdahl p=0.90', linewidth=1.5)
    ax1.plot(workers, amdahl_95, ':', color='#e67e22', label='Amdahl p=0.95', linewidth=1.5)
    ax1.plot(workers, amdahl_99, ':', color='#d35400', label='Amdahl p=0.99', linewidth=1.5)
    ax1.plot(workers, sp_sin, 'o-', color='#2ecc71', label='Sin overhead', linewidth=2)
    ax1.plot(workers, sp_con, 's-', color='#e74c3c', label='Con overhead coord.', linewidth=2)
    ax1.set_xlabel("Nº de workers"); ax1.set_ylabel("Speedup (×)")
    ax1.set_title("Speedup real vs Ley de Amdahl"); ax1.legend(fontsize=7); ax1.grid(alpha=0.3)

    # ── Gráfico 2: Eficiencia ──
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(workers, ef_sin, 'o-', color='#2ecc71', label='Sin overhead', linewidth=2)
    ax2.plot(workers, ef_con, 's-', color='#e74c3c', label='Con overhead', linewidth=2)
    ax2.axhline(1.0, color='gray', linestyle='--', linewidth=1, label='Eficiencia ideal')
    ax2.set_xlabel("Nº de workers"); ax2.set_ylabel("Eficiencia (Speedup/workers)")
    ax2.set_title("Eficiencia del paralelismo"); ax2.legend(); ax2.grid(alpha=0.3)

    # ── Gráfico 3: Tiempos absolutos ──
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(workers, t_sin, 'o-', color='#2ecc71', label='Sin overhead', linewidth=2)
    ax3.plot(workers, t_con, 's-', color='#e74c3c', label='Con overhead', linewidth=2)
    ax3.axhline(m_sec, color='#3498db', linestyle='--', linewidth=1.5, label=f'Secuencial ({m_sec:.3f}s)')
    ax3.set_xlabel("Nº de workers"); ax3.set_ylabel("Tiempo (s)")
    ax3.set_title("Tiempo de ejecución"); ax3.legend(); ax3.grid(alpha=0.3)

    # ── Gráfico 4: Degradación del speedup (%) ──
    ax4 = fig.add_subplot(gs[1, 0])
    degradacion = [(sp_sin[i] - sp_con[i]) / sp_sin[i] * 100 for i in range(len(workers))]
    colores_deg = ['#e74c3c' if d > 0 else '#2ecc71' for d in degradacion]
    ax4.bar([str(w) for w in workers], degradacion, color=colores_deg, alpha=0.85)
    ax4.set_xlabel("Nº de workers"); ax4.set_ylabel("Degradación (%)")
    ax4.set_title("% de speedup perdido por overhead")
    ax4.grid(alpha=0.3, axis='y')
    for i, d in enumerate(degradacion):
        ax4.text(i, d + 0.5, f"{d:.1f}%", ha='center', va='bottom', fontsize=8)

    # ── Gráfico 5: Recomendación técnica por escenario ──
    ax5 = fig.add_subplot(gs[1, 1:])
    ax5.axis('off')

    escenarios = [
        {
            "nombre": "🖥️  Servidor Único",
            "descripcion": (
                "Datos caben en RAM.\n"
                "Tarea intensiva en CPU (sin I/O dominante).\n"
                "→ Recomendación: NumPy vectorizado +\n"
                "  ProcessPoolExecutor (≤ núcleos físicos).\n"
                "  Dask local solo si datos > RAM.\n"
                "  Evitar ThreadPool en tareas CPU-bound."
            ),
            "color": "#d5e8d4"
        },
        {
            "nombre": "🏢  Org. Multi-fuente de Datos",
            "descripcion": (
                "Datos en múltiples sistemas heterogéneos.\n"
                "Pipelines con joins y agregaciones complejas.\n"
                "→ Recomendación: Dask o PySpark (local/cluster)\n"
                "  con particionamiento por fuente.\n"
                "  Tolera overhead de coordinación por\n"
                "  ganancia en paralelismo de datos."
            ),
            "color": "#dae8fc"
        },
        {
            "nombre": "☁️  Plataforma Cloud",
            "descripcion": (
                "Datos masivos (TB+), escalabilidad elástica.\n"
                "Latencia de red y costo son factores clave.\n"
                "→ Recomendación: Spark/Dask distribuido\n"
                "  con particionamiento optimizado.\n"
                "  Escalar horizontalmente según carga.\n"
                "  Monitorear overhead de coordinación."
            ),
            "color": "#ffe6cc"
        }
    ]

    x_pos = [0.02, 0.36, 0.68]
    for i, (esc, xp) in enumerate(zip(escenarios, x_pos)):
        rect = mpatches.FancyBboxPatch((xp, 0.05), 0.3, 0.9,
                                       boxstyle="round,pad=0.02",
                                       facecolor=esc["color"],
                                       edgecolor='#666',
                                       linewidth=1.2,
                                       transform=ax5.transAxes)
        ax5.add_patch(rect)
        ax5.text(xp + 0.15, 0.90, esc["nombre"],
                 ha='center', va='top', fontsize=10, fontweight='bold',
                 transform=ax5.transAxes)
        ax5.text(xp + 0.15, 0.78, esc["descripcion"],
                 ha='center', va='top', fontsize=8.2,
                 transform=ax5.transAxes, linespacing=1.4)

    ax5.set_title("Decisión Técnica por Escenario de Infraestructura",
                  fontsize=11, fontweight='bold')

    plt.savefig("/mnt/user-data/outputs/experimento_D.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("\n[✓] Gráfico guardado: experimento_D.png")




def tabla_resumen_D(resultados, m_sec):
    filas = []
    for nw, sp_s, ef_s, t_s, sp_o, ef_o, t_o in zip(
        WORKERS_RANGE,
        resultados["sin_overhead"]["speedup"],
        resultados["sin_overhead"]["eficiencia"],
        resultados["sin_overhead"]["tiempo"],
        resultados["con_overhead"]["speedup"],
        resultados["con_overhead"]["eficiencia"],
        resultados["con_overhead"]["tiempo"],
    ):
        filas.append({
            "Workers": nw,
            "t_sin_OH (s)": round(t_s, 4),
            "Speedup sin OH": round(sp_s, 3),
            "Efic. sin OH": round(ef_s, 3),
            "t_con_OH (s)": round(t_o, 4),
            "Speedup con OH": round(sp_o, 3),
            "Efic. con OH": round(ef_o, 3),
        })
    df = pd.DataFrame(filas)
    print("\n[Tabla resumen Experimento D]")
    print(df.to_string(index=False))
    return df


if __name__ == "__main__":
    resultados, m_sec, datos = ejecutar_experimento_D()
    tabla_resumen_D(resultados, m_sec)
    graficar_D(resultados, m_sec)
