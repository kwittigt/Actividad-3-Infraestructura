"""
Script principal: ejecuta los 4 experimentos en orden y genera
un dashboard resumen de todas las visualizaciones.

Uso:
    python main.py

Requisitos:
    pip install numpy pandas matplotlib seaborn dask psutil joblib pyarrow
"""

import sys
import os
import platform
import time
import psutil
import multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime




def registrar_entorno():
    print("\n" + "═" * 65)
    print("  REGISTRO DEL ENTORNO EXPERIMENTAL")
    print("═" * 65)
    print(f"  Fecha/hora       : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Sistema operativo: {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"  Python           : {sys.version.split()[0]}")
    print(f"  CPU              : {platform.processor() or 'N/D'}")
    print(f"  Núcleos lógicos  : {mp.cpu_count()}")
    print(f"  Núcleos físicos  : {psutil.cpu_count(logical=False)}")
    print(f"  RAM total        : {psutil.virtual_memory().total / 1e9:.2f} GB")
    print(f"  RAM disponible   : {psutil.virtual_memory().available / 1e9:.2f} GB")

    # Versiones de librerías
    librerias = ["numpy", "pandas", "matplotlib", "dask", "psutil", "joblib"]
    print("\n  Librerías instaladas:")
    for lib in librerias:
        try:
            mod = __import__(lib)
            ver = getattr(mod, "__version__", "?")
        except ImportError:
            ver = "NO INSTALADA"
        print(f"    {lib:15s}: {ver}")

    # GPU
    try:
        import subprocess
        result = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                                 "--format=csv,noheader"],
                                capture_output=True, text=True, timeout=3)
        if result.returncode == 0:
            print(f"\n  GPU (NVIDIA)     : {result.stdout.strip()}")
        else:
            print("\n  GPU              : No disponible (sin GPU NVIDIA detectada)")
    except Exception:
        print("\n  GPU              : No disponible (sin GPU NVIDIA detectada)")

    print("═" * 65 + "\n")




def ejecutar_todos():
    t_inicio_total = time.perf_counter()

    registrar_entorno()
    os.makedirs("/mnt/user-data/outputs", exist_ok=True)


    print("\n" + "▶" * 3 + " EJECUTANDO EXPERIMENTO A...")
    t0 = time.perf_counter()
    from experimento_A import ejecutar_experimento_A, graficar_A
    res_A = ejecutar_experimento_A()
    graficar_A(res_A)
    print(f"  [Exp A completado en {time.perf_counter()-t0:.1f}s]")

  
    print("\n" + "▶" * 3 + " EJECUTANDO EXPERIMENTO B...")
    t0 = time.perf_counter()
    from experimento_B import ejecutar_experimento_B, graficar_B
    res_B = ejecutar_experimento_B()
    graficar_B(res_B)
    print(f"  [Exp B completado en {time.perf_counter()-t0:.1f}s]")

  
    print("\n" + "▶" * 3 + " EJECUTANDO EXPERIMENTO C...")
    t0 = time.perf_counter()
    from experimento_C import ejecutar_experimento_C, graficar_C
    res_C, m_pd = ejecutar_experimento_C()
    graficar_C(res_C, m_pd)
    print(f"  [Exp C completado en {time.perf_counter()-t0:.1f}s]")

  
    print("\n" + "▶" * 3 + " EJECUTANDO EXPERIMENTO D...")
    t0 = time.perf_counter()
    from experimento_D import ejecutar_experimento_D, tabla_resumen_D, graficar_D
    res_D, m_sec, _ = ejecutar_experimento_D()
    tabla_resumen_D(res_D, m_sec)
    graficar_D(res_D, m_sec)
    print(f"  [Exp D completado en {time.perf_counter()-t0:.1f}s]")

    
    generar_dashboard()

    t_total = time.perf_counter() - t_inicio_total
    print(f"\n{'═'*65}")
    print(f"  ✅ Todos los experimentos completados en {t_total:.1f}s")
    print(f"  📁 Archivos generados en: /mnt/user-data/outputs/")
    print(f"{'═'*65}\n")


def generar_dashboard():
    """Combina los 4 gráficos en un dashboard resumen."""
    imagenes = [
        "/mnt/user-data/outputs/experimento_A.png",
        "/mnt/user-data/outputs/experimento_B.png",
        "/mnt/user-data/outputs/experimento_C.png",
        "/mnt/user-data/outputs/experimento_D.png",
    ]
    titulos = [
        "A – Vectorización y Paralelismo de Datos",
        "B – Paralelismo a Nivel de Tareas",
        "C – Procesamiento Distribuido (Dask)",
        "D – Overhead de Coordinación",
    ]

    fig, axes = plt.subplots(2, 2, figsize=(20, 14))
    fig.suptitle(
        "INFB6074 – Actividad Semana 3\nDashboard: Paralelismo Local y Procesamiento Distribuido",
        fontsize=16, fontweight='bold', y=0.98
    )

    for ax, img_path, titulo in zip(axes.flat, imagenes, titulos):
        if os.path.exists(img_path):
            img = plt.imread(img_path)
            ax.imshow(img)
        else:
            ax.text(0.5, 0.5, f"[{titulo}\n – no disponible]",
                    ha='center', va='center', transform=ax.transAxes)
        ax.set_title(titulo, fontsize=11, fontweight='bold', pad=6)
        ax.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path_dashboard = "/mnt/user-data/outputs/dashboard_resumen.png"
    plt.savefig(path_dashboard, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"\n[✓] Dashboard guardado: dashboard_resumen.png")



if __name__ == "__main__":
    # Necesario en Windows/macOS para evitar fork recursivo en multiprocessing
    mp.freeze_support()
    ejecutar_todos()
