"""
evaluacion.py - Fase 5: Comparación entre la solución heurística y la solución MLP.

Métricas de comparación:
  1. Turnos cubiertos (cobertura)
  2. Distribución de horas (equilibrio - desviación estándar)
  3. Violaciones de restricciones
  4. Utilización total de horas
"""
import json
import os
import numpy as np

from src.cargar_datos import cargar_empleados, cargar_turnos
from src.restricciones import validar_asignacion_completa

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTADOS_DIR = os.path.join(BASE_DIR, "resultados")


def comparar_soluciones(resultado_heur, resultado_mlp, empleados, turnos):
    """
    Compara las dos soluciones y genera un reporte detallado.
    
    Returns:
        dict con comparación completa
    """
    print("\n" + "=" * 60)
    print("  FASE 5: COMPARACIÓN DE SOLUCIONES")
    print("=" * 60)

    # Extraer horas por empleado
    horas_heur = resultado_heur["resumen"]["horas_por_empleado"]
    horas_mlp = resultado_mlp["resumen"]["horas_por_empleado"]

    # Calcular métricas
    horas_heur_list = list(horas_heur.values())
    horas_mlp_list = list(horas_mlp.values())

    metricas = {
        "heuristica": {
            "metodo": "Heurística Greedy",
            "turnos_cubiertos": resultado_heur["resumen"]["turnos_cubiertos"],
            "turnos_total": resultado_heur["resumen"]["turnos_total"],
            "cobertura_pct": round(resultado_heur["resumen"]["turnos_cubiertos"] / resultado_heur["resumen"]["turnos_total"] * 100, 1),
            "horas_totales": sum(horas_heur_list),
            "horas_promedio": round(np.mean(horas_heur_list), 1),
            "horas_std": round(np.std(horas_heur_list), 2),
            "horas_min": min(horas_heur_list),
            "horas_max": max(horas_heur_list),
            "horas_por_empleado": horas_heur
        },
        "mlp": {
            "metodo": "Red Neuronal MLP",
            "turnos_cubiertos": resultado_mlp["resumen"]["turnos_cubiertos"],
            "turnos_total": resultado_mlp["resumen"]["turnos_total"],
            "cobertura_pct": round(resultado_mlp["resumen"]["turnos_cubiertos"] / resultado_mlp["resumen"]["turnos_total"] * 100, 1),
            "horas_totales": sum(horas_mlp_list),
            "horas_promedio": round(np.mean(horas_mlp_list), 1),
            "horas_std": round(np.std(horas_mlp_list), 2),
            "horas_min": min(horas_mlp_list),
            "horas_max": max(horas_mlp_list),
            "horas_por_empleado": horas_mlp
        }
    }

    # Determinar ganador por métrica
    ganadores = {}
    if metricas["heuristica"]["turnos_cubiertos"] > metricas["mlp"]["turnos_cubiertos"]:
        ganadores["cobertura"] = "Heurística"
    elif metricas["mlp"]["turnos_cubiertos"] > metricas["heuristica"]["turnos_cubiertos"]:
        ganadores["cobertura"] = "MLP"
    else:
        ganadores["cobertura"] = "Empate"

    if metricas["heuristica"]["horas_std"] < metricas["mlp"]["horas_std"]:
        ganadores["equilibrio"] = "Heurística"
    elif metricas["mlp"]["horas_std"] < metricas["heuristica"]["horas_std"]:
        ganadores["equilibrio"] = "MLP"
    else:
        ganadores["equilibrio"] = "Empate"

    if metricas["heuristica"]["horas_totales"] > metricas["mlp"]["horas_totales"]:
        ganadores["utilizacion"] = "Heurística"
    elif metricas["mlp"]["horas_totales"] > metricas["heuristica"]["horas_totales"]:
        ganadores["utilizacion"] = "MLP"
    else:
        ganadores["utilizacion"] = "Empate"

    comparacion = {
        "metricas": metricas,
        "ganadores": ganadores,
        "asignaciones_heuristica": resultado_heur["asignaciones"],
        "asignaciones_mlp": resultado_mlp["asignaciones"]
    }

    # Imprimir tabla comparativa
    print(f"\n  {'Métrica':<25} {'Heurística':>15} {'MLP':>15} {'Ganador':>15}")
    print("  " + "-" * 70)
    print(f"  {'Turnos cubiertos':<25} {metricas['heuristica']['turnos_cubiertos']}/{metricas['heuristica']['turnos_total']:>13} {metricas['mlp']['turnos_cubiertos']}/{metricas['mlp']['turnos_total']:>13} {ganadores['cobertura']:>15}")
    print(f"  {'Cobertura %':<25} {metricas['heuristica']['cobertura_pct']:>14}% {metricas['mlp']['cobertura_pct']:>14}%")
    print(f"  {'Horas totales':<25} {metricas['heuristica']['horas_totales']:>15} {metricas['mlp']['horas_totales']:>15} {ganadores['utilizacion']:>15}")
    print(f"  {'Horas promedio':<25} {metricas['heuristica']['horas_promedio']:>15} {metricas['mlp']['horas_promedio']:>15}")
    print(f"  {'Desv. Estándar (eq.)':<25} {metricas['heuristica']['horas_std']:>15} {metricas['mlp']['horas_std']:>15} {ganadores['equilibrio']:>15}")
    print(f"  {'Horas mín':<25} {metricas['heuristica']['horas_min']:>15} {metricas['mlp']['horas_min']:>15}")
    print(f"  {'Horas máx':<25} {metricas['heuristica']['horas_max']:>15} {metricas['mlp']['horas_max']:>15}")

    print(f"\n  Distribución por empleado:")
    for emp in empleados:
        eid = emp["id"]
        h_h = horas_heur.get(eid, 0)
        h_m = horas_mlp.get(eid, 0)
        print(f"    {eid} {emp['nombre']:8s} | Heur: {h_h:2d}h | MLP: {h_m:2d}h | Max: {emp['horas_max_semana']}h")

    # Guardar comparación
    os.makedirs(RESULTADOS_DIR, exist_ok=True)
    ruta = os.path.join(RESULTADOS_DIR, "comparacion.json")
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(comparacion, f, ensure_ascii=False, indent=2)
    print(f"\n[GUARDADO] Comparación → {ruta}")

    return comparacion
