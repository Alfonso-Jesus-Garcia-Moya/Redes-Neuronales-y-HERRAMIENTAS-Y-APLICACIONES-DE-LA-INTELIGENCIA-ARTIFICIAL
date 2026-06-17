"""
heuristica.py - Solución base (Fase 3): Algoritmo Greedy para asignación de turnos.

Estrategia:
  Para cada turno, seleccionar empleados disponibles priorizando
  a quienes tengan MENOS horas asignadas (distribución equilibrada).
"""
import json
import os

from src.cargar_datos import cargar_empleados, cargar_turnos
from src.restricciones import esta_disponible, puede_trabajar_mas, ya_trabaja_ese_dia

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTADOS_DIR = os.path.join(BASE_DIR, "resultados")


def ejecutar_heuristica():
    """
    Ejecuta el algoritmo greedy de asignación de turnos.
    
    Retorna:
        dict con asignaciones y resumen
    """
    empleados = cargar_empleados()
    turnos = cargar_turnos()

    # Estructuras de seguimiento
    asignaciones = {}           # {turno_id: [emp_id, ...]}
    horas_asignadas = {}        # {emp_id: horas_totales}

    for emp in empleados:
        horas_asignadas[emp["id"]] = 0

    print("\n" + "=" * 60)
    print("  EJECUTANDO HEURÍSTICA GREEDY")
    print("=" * 60)

    for turno in turnos:
        turno_id = turno["id"]
        requeridos = turno["empleados_requeridos"]

        # Paso 1: Filtrar candidatos válidos
        candidatos = []
        for emp in empleados:
            emp_id = emp["id"]

            # ¿Está disponible para este turno?
            if not esta_disponible(emp, turno):
                continue

            # ¿Puede trabajar más horas?
            if not puede_trabajar_mas(emp_id, asignaciones, empleados, turno["horas"]):
                continue

            # ¿Ya trabaja ese día?
            if ya_trabaja_ese_dia(emp_id, turno["dia"], asignaciones, turnos):
                continue

            candidatos.append(emp)

        # Paso 2: Ordenar por horas asignadas (ascendente) → equilibrio
        candidatos.sort(key=lambda e: horas_asignadas[e["id"]])

        # Paso 3: Asignar los primeros N candidatos
        seleccionados = candidatos[:requeridos]
        asignaciones[turno_id] = [e["id"] for e in seleccionados]

        # Paso 4: Actualizar horas
        for emp in seleccionados:
            horas_asignadas[emp["id"]] += turno["horas"]

        # Log
        nombres = [e["nombre"] for e in seleccionados]
        estado = "✓" if len(seleccionados) == requeridos else "✗ INCOMPLETO"
        print(f"  {turno_id} | {turno['dia']:10s} {turno['horario']:10s} | "
              f"Asignados: {nombres} | {estado}")

    # Construir resultado
    asignaciones_detalle = []
    for turno in turnos:
        asignaciones_detalle.append({
            "turno_id": turno["id"],
            "dia": turno["dia"],
            "horario": turno["horario"],
            "empleados_asignados": asignaciones.get(turno["id"], [])
        })

    resultado = {
        "metodo": "heuristica_greedy",
        "asignaciones": asignaciones_detalle,
        "resumen": {
            "turnos_cubiertos": sum(
                1 for t in turnos
                if len(asignaciones.get(t["id"], [])) >= t["empleados_requeridos"]
            ),
            "turnos_total": len(turnos),
            "horas_por_empleado": horas_asignadas
        }
    }

    # Guardar resultado en JSON
    os.makedirs(RESULTADOS_DIR, exist_ok=True)
    ruta_salida = os.path.join(RESULTADOS_DIR, "asignacion_heuristica.json")
    with open(ruta_salida, "w", encoding="utf-8") as f:
        json.dump(resultado, f, ensure_ascii=False, indent=2)
    print(f"\n[GUARDADO] Resultado heurística → {ruta_salida}")

    # Resumen en consola
    print("\n--- RESUMEN HEURÍSTICA ---")
    print(f"  Turnos cubiertos: {resultado['resumen']['turnos_cubiertos']}/{resultado['resumen']['turnos_total']}")
    print(f"  Horas por empleado:")
    for emp in empleados:
        h = horas_asignadas[emp["id"]]
        barra = "█" * (h // 5) + "░" * ((emp["horas_max_semana"] - h) // 5)
        print(f"    {emp['id']} {emp['nombre']:8s} | {h:2d}h / {emp['horas_max_semana']}h max | {barra}")

    return resultado, asignaciones


if __name__ == "__main__":
    ejecutar_heuristica()
