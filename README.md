### Estructura del repositorio

```
actividad_semana3/
├── main.py               ← Script principal (ejecuta todo)
├── experimento_A.py      ← Exp A: Vectorización y paralelismo de datos
├── experimento_B.py      ← Exp B: Paralelismo a nivel de tareas
├── experimento_C.py      ← Exp C: Procesamiento distribuido con Dask
├── experimento_D.py      ← Exp D: Overhead de coordinación
├── README.md             ← Este archivo
└── outputs/
    ├── experimento_A.png
    ├── experimento_B.png
    ├── experimento_C.png
    ├── experimento_D.png
    └── dashboard_resumen.png
```

---

### Requisitos

- Python 3.9+
- Librerías:

```bash
pip install numpy pandas matplotlib seaborn dask psutil joblib pyarrow
```

---

### Ejecución

#### Opción 1: Ejecutar todos los experimentos

```bash
python main.py
```

#### Opción 2: Ejecutar un experimento individual

```bash
python experimento_A.py
python experimento_B.py
python experimento_C.py
python experimento_D.py
```

---

### Descripción de cada experimento

| Script            | Experimento | Descripción |
|-------------------|-------------|-------------|
| `experimento_A.py`| A           | Compara Python puro, NumPy vectorizado y `multiprocessing` sobre arreglos grandes. Mide speedup y eficiencia en 5 tamaños. |
| `experimento_B.py`| B           | Compara ejecución secuencial, `ThreadPoolExecutor` y `ProcessPoolExecutor` para feature engineering en lotes. Varía workers y tamaño de chunk. |
| `experimento_C.py`| C           | Compara Pandas monolítico contra Dask DataFrame con 3 configuraciones de particionamiento en un pipeline ETL de 2M filas. |
| `experimento_D.py`| D           | Modela overhead de serialización/coordinación y muestra por qué el speedup no escala linealmente. Incluye recomendaciones técnicas por escenario. |

---

### Parámetros configurables

Cada script tiene una sección `Configuración` al inicio donde se pueden modificar:

- `N_REPETICIONES`: cantidad de repeticiones para promediar (defecto: 3)
- `TAMANIOS`: tamaños de entrada para Experimento A
- `N_LOTES`, `TAMANIOS_LOTE`: configuración de lotes en Experimento B
- `PARTICIONES_DASK`: número de particiones en Experimento C
- `WORKERS_RANGE`: rango de workers en Experimento D

---

### Nota sobre GPU

Este experimento no incluye comparación GPU ya que el entorno no dispone de CUDA.
La comparación equivalente se realiza entre:
- Vectorización NumPy (CPU vectorizado)
- Procesamiento multicore local
- Procesamiento distribuido local (Dask)

---

### Declaración de uso de herramientas

Los experimentos y el análisis fueron desarrollados con apoyo de herramientas de IA como asistente de diseño. El diseño experimental, los parámetros de prueba, la interpretación y las conclusiones son propios del grupo.
