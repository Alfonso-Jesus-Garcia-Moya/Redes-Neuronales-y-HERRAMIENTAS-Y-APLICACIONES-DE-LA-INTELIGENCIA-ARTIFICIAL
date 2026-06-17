"""
generar_datos_entrenamiento.py - Fase 4: Generación de dataset sintético
para entrenar la red neuronal MLP.

Estrategia:
  1. Generar múltiples soluciones válidas usando la heurística con
     orden aleatorio de turnos y empleados (variación estocástica)
  2. Para cada solución, crear pares (features, label) donde:
     - features = vector numérico que describe un par (empleado, turno)
     - label = 1 si el empleado fue asignado a ese turno, 0 si no
  3. Guardar el dataset en data/dataset_entrenamiento.json
"""
import json
import os
import random
import numpy as np

from src.cargar_datos import cargar_empleados, cargar_turnos
from src.restricciones import esta_disponible, puede_trabajar_mas, ya_trabaja_ese_dia

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

# Mapeos para codificación numérica
DIAS_MAP = {
    "Lunes": 0, "Martes": 1, "Miércoles": 2, "Jueves": 3,
    "Viernes": 4, "Sábado": 5, "Domingo": 6
}
HORARIOS_MAP = {"Mañana": 0, "Tarde": 1, "Completo": 2}


def codificar_features(empleado, turno, horas_ya_asignadas, turnos_dia_ya):
    """
    Convierte un par (empleado, turno) en un vector numérico de features.
    
    Features (10 dimensiones):
      0: horas_max_semana (normalizado /40)
      1: horas_ya_asignadas (normalizado /40)
      2: horas_restantes (normalizado /40)
      3: está_disponible (0 o 1)
      4: día del turno (normalizado /6)
      5: horario del turno (0=mañana, 0.5=tarde, 1=completo)
      6: empleados_requeridos del turno (normalizado /3)
      7: ya_trabaja_ese_dia (0 o 1)
      8: ratio horas usadas vs max (0 a 1)
      9: número de días disponibles del empleado (normalizado /7)
    """
    horas_max = empleado["horas_max_semana"]
    disponible = 1.0 if esta_disponible(empleado, turno) else 0.0
    dia_num = DIAS_MAP.get(turno["dia"], 0) / 6.0
    horario_num = HORARIOS_MAP.get(turno["horario"], 0) / 2.0
    horas_restantes = max(0, horas_max - horas_ya_asignadas)
    ratio_uso = horas_ya_asignadas / horas_max if horas_max > 0 else 1.0
    dias_disponibles = len(empleado["disponibilidad"]) / 7.0
    ya_trabaja = 1.0 if turnos_dia_ya else 0.0

    return [
        horas_max / 40.0,           # 0
        horas_ya_asignadas / 40.0,  # 1
        horas_restantes / 40.0,     # 2
        disponible,                 # 3
        dia_num,                    # 4
        horario_num,                # 5
        turno["empleados_requeridos"] / 3.0,  # 6
        ya_trabaja,                 # 7
        ratio_uso,                  # 8
        dias_disponibles            # 9
    ]


def generar_solucion_aleatoria(empleados, turnos):
    """
    Genera una solución usando heurística greedy pero con orden
    aleatorio de turnos y desempate aleatorio entre candidatos.
    """
    turnos_shuffled = turnos.copy()
    random.shuffle(turnos_shuffled)

    asignaciones = {}
    horas_asignadas = {emp["id"]: 0 for emp in empleados}

    for turno in turnos_shuffled:
        turno_id = turno["id"]
        requeridos = turno["empleados_requeridos"]

        candidatos = []
        for emp in empleados:
            emp_id = emp["id"]
            if not esta_disponible(emp, turno):
                continue
            if not puede_trabajar_mas(emp_id, asignaciones, empleados, turno["horas"]):
                continue
            if ya_trabaja_ese_dia(emp_id, turno["dia"], asignaciones, turnos):
                continue
            candidatos.append(emp)

        # Ordenar por horas (equilibrio) + ruido aleatorio para variación
        candidatos.sort(key=lambda e: horas_asignadas[e["id"]] + random.uniform(0, 3))

        seleccionados = candidatos[:requeridos]
        asignaciones[turno_id] = [e["id"] for e in seleccionados]

        for emp in seleccionados:
            horas_asignadas[emp["id"]] += turno["horas"]

    # Calcular score de la solución
    turnos_cubiertos = sum(
        1 for t in turnos
        if len(asignaciones.get(t["id"], [])) >= t["empleados_requeridos"]
    )
    horas_list = list(horas_asignadas.values())
    balance = np.std(horas_list) if horas_list else 0
    score = turnos_cubiertos * 10 - balance

    return asignaciones, horas_asignadas, score


def generar_dataset(n_soluciones=500):
    """
    Genera el dataset de entrenamiento.
    
    Args:
        n_soluciones: Número de soluciones aleatorias a generar
    
    Returns:
        tuple (X, y) donde X son features y y son labels
    """
    empleados = cargar_empleados()
    turnos = cargar_turnos()

    print(f"\n[DATASET] Generando {n_soluciones} soluciones aleatorias...")

    todas_X = []
    todas_y = []
    mejores_scores = []

    for i in range(n_soluciones):
        asignaciones, horas_asignadas, score = generar_solucion_aleatoria(empleados, turnos)
        mejores_scores.append(score)

        # Para cada par (empleado, turno), generar features y label
        for turno in turnos:
            for emp in empleados:
                emp_id = emp["id"]

                # Calcular si ya trabaja ese día en esta solución
                turnos_dia = ya_trabaja_ese_dia(emp_id, turno["dia"], asignaciones, turnos)

                features = codificar_features(
                    emp, turno,
                    horas_asignadas.get(emp_id, 0),
                    turnos_dia
                )

                # Label: 1 si fue asignado, 0 si no
                asignados = asignaciones.get(turno["id"], [])
                label = 1.0 if emp_id in asignados else 0.0

                todas_X.append(features)
                todas_y.append(label)

    X = np.array(todas_X, dtype=np.float32)
    y = np.array(todas_y, dtype=np.float32)

    print(f"[DATASET] Dataset generado: {X.shape[0]} muestras, {X.shape[1]} features")
    print(f"[DATASET] Distribución labels: {np.sum(y == 1):.0f} positivos, {np.sum(y == 0):.0f} negativos")
    print(f"[DATASET] Score promedio de soluciones: {np.mean(mejores_scores):.2f}")
    print(f"[DATASET] Mejor score: {np.max(mejores_scores):.2f}")

    # Guardar dataset
    dataset_path = os.path.join(DATA_DIR, "dataset_entrenamiento.json")
    dataset = {
        "X": X.tolist(),
        "y": y.tolist(),
        "n_features": X.shape[1],
        "n_muestras": X.shape[0],
        "n_soluciones": n_soluciones,
        "feature_names": [
            "horas_max_norm", "horas_asignadas_norm", "horas_restantes_norm",
            "esta_disponible", "dia_norm", "horario_norm",
            "empleados_requeridos_norm", "ya_trabaja_dia",
            "ratio_uso", "dias_disponibles_norm"
        ]
    }
    with open(dataset_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False)
    print(f"[DATASET] Guardado en → {dataset_path}")

    return X, y


if __name__ == "__main__":
    X, y = generar_dataset(500)
