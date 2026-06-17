"""
cargar_datos.py - Carga y validación de los JSONs de empleados y turnos.
"""
import json
import os

# Ruta base del proyecto (scheduling_mvp/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")


def cargar_empleados():
    """Lee data/empleados.json y retorna la lista de diccionarios."""
    ruta = os.path.join(DATA_DIR, "empleados.json")
    with open(ruta, "r", encoding="utf-8") as f:
        empleados = json.load(f)
    print(f"[DATOS] Se cargaron {len(empleados)} empleados.")
    return empleados


def cargar_turnos():
    """Lee data/turnos.json y retorna la lista de diccionarios."""
    ruta = os.path.join(DATA_DIR, "turnos.json")
    with open(ruta, "r", encoding="utf-8") as f:
        turnos = json.load(f)
    print(f"[DATOS] Se cargaron {len(turnos)} turnos.")
    return turnos


if __name__ == "__main__":
    empleados = cargar_empleados()
    turnos = cargar_turnos()
    print("\n--- Empleados ---")
    for e in empleados:
        print(f"  {e['id']} - {e['nombre']} | Max {e['horas_max_semana']}h | Días: {list(e['disponibilidad'].keys())}")
    print("\n--- Turnos ---")
    for t in turnos:
        print(f"  {t['id']} - {t['dia']} {t['horario']} | Requiere: {t['empleados_requeridos']} empleados")
