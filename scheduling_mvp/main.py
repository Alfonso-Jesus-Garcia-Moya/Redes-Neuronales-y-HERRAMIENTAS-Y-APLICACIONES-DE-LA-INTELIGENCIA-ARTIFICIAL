"""
main.py - Orquestador principal del Proyecto Integrador:
         Optimización de Asignación de Turnos con IA y Redes Neuronales

Ejecutar todo:     python main.py
Solo dashboard:    python main.py --dashboard
"""
import sys
import os

# Agregar la raíz del proyecto al path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.cargar_datos import cargar_empleados, cargar_turnos
from src.heuristica import ejecutar_heuristica
from src.restricciones import validar_asignacion_completa


def main():
    print("\n" + "=" * 60)
    print("  PROYECTO INTEGRADOR: SCHEDULING CON IA")
    print("  Optimización de Asignación de Turnos")
    print("  mediante Inteligencia Artificial y Redes Neuronales")
    print("=" * 60)

    # =========================================================
    # FASE 1-2: Carga de datos y modelado
    # =========================================================
    print("\n[FASE 1-2] Cargando datos y modelado del problema...")
    empleados = cargar_empleados()
    turnos = cargar_turnos()

    # =========================================================
    # FASE 3: Solución base - Heurística Greedy
    # =========================================================
    print("\n[FASE 3] Ejecutando solución base (Heurística Greedy)...")
    resultado_heuristica, asignaciones_heur = ejecutar_heuristica()

    # Validar la solución heurística
    print("\n[VALIDACIÓN] Verificando restricciones de la solución heurística...")
    reporte_heur = validar_asignacion_completa(asignaciones_heur, turnos, empleados)
    imprimir_reporte(reporte_heur, empleados, "HEURÍSTICA")

    # =========================================================
    # FASE 4: Red Neuronal MLP
    # =========================================================
    print("\n[FASE 4] Generando dataset y entrenando modelo MLP...")
    from src.generar_datos_entrenamiento import generar_dataset
    from src.modelo_mlp import entrenar_mlp, predecir_asignacion, guardar_historial

    X, y = generar_dataset(500)
    modelo, historial = entrenar_mlp(X, y, epochs=50, batch_size=32)
    guardar_historial(historial)
    resultado_mlp, asignaciones_mlp = predecir_asignacion(modelo, empleados, turnos)

    # Validar la solución MLP
    print("\n[VALIDACIÓN] Verificando restricciones de la solución MLP...")
    reporte_mlp = validar_asignacion_completa(asignaciones_mlp, turnos, empleados)
    imprimir_reporte(reporte_mlp, empleados, "MLP")

    # =========================================================
    # FASE 5: Evaluación comparativa
    # =========================================================
    print("\n[FASE 5] Comparando soluciones...")
    from src.evaluacion import comparar_soluciones
    comparacion = comparar_soluciones(resultado_heuristica, resultado_mlp, empleados, turnos)

    # =========================================================
    # FASE 6: Visualización Flask
    # =========================================================
    print("\n" + "=" * 60)
    print("  TODAS LAS FASES COMPLETADAS")
    print("  Iniciando visualización en localhost...")
    print("=" * 60)

    from src.app import iniciar_app
    iniciar_app()


def imprimir_reporte(reporte, empleados, nombre):
    """Imprime el reporte de validación de una solución."""
    print(f"\n  --- Reporte {nombre} ---")
    print(f"  Turnos cubiertos: {reporte['turnos_cubiertos']}/{reporte['turnos_total']}")
    print(f"  Solución válida: {'✓ SÍ' if reporte['es_valida'] else '✗ NO'}")
    print(f"  Total violaciones: {reporte['total_violaciones']}")

    if reporte["turnos_sin_cubrir"]:
        print(f"\n  ⚠ Turnos sin cubrir:")
        for t in reporte["turnos_sin_cubrir"]:
            print(f"    - {t['turno']}: {t['asignados']}/{t['requeridos']} empleados")

    if reporte["violaciones_horas"]:
        print(f"\n  ⚠ Violaciones de horas máximas:")
        for v in reporte["violaciones_horas"]:
            print(f"    - {v['nombre']}: {v['horas_asignadas']}h (max {v['horas_max']}h)")

    print(f"\n  Distribución de horas:")
    for emp_id, horas in reporte["horas_por_empleado"].items():
        emp = next(e for e in empleados if e["id"] == emp_id)
        print(f"    {emp_id} {emp['nombre']:8s}: {horas}h / {emp['horas_max_semana']}h max")


def solo_dashboard():
    """Inicia solo el dashboard Flask sin re-ejecutar modelos."""
    print("\n[DASHBOARD] Iniciando solo visualización...")
    print("[DASHBOARD] Usando resultados previamente generados.\n")
    from src.app import iniciar_app
    iniciar_app()


if __name__ == "__main__":
    if "--dashboard" in sys.argv:
        solo_dashboard()
    else:
        main()
