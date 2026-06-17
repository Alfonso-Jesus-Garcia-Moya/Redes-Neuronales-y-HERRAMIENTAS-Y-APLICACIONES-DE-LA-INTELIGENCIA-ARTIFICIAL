"""
restricciones.py - Funciones de validación de restricciones del scheduling.

Restricciones del problema:
  1. Un empleado solo puede asignarse si está disponible en ese día/horario
  2. No puede exceder sus horas máximas semanales
  3. Máximo 1 turno por día por empleado
  4. Cada turno debe tener exactamente N empleados requeridos
  5. Distribución equilibrada de horas
"""


def esta_disponible(empleado, turno):
    """
    Verifica si un empleado está disponible para un turno específico.
    
    Lógica de compatibilidad de horarios:
    - Si el empleado tiene 'Completo' → cubre 'Mañana', 'Tarde' y 'Completo'
    - Si el turno es 'Completo' → el empleado necesita 'Completo' O tener
      tanto 'Mañana' como 'Tarde' ese día
    - Si el turno es 'Mañana'/'Tarde' → el empleado necesita ese horario
      O 'Completo'
    """
    dia_turno = turno["dia"]
    horario_turno = turno["horario"]
    disponibilidad = empleado.get("disponibilidad", {})

    # Si el empleado no tiene disponibilidad ese día → No disponible
    if dia_turno not in disponibilidad:
        return False

    horarios_empleado = disponibilidad[dia_turno]

    # Caso 1: El turno es 'Completo'
    if horario_turno == "Completo":
        # El empleado necesita tener 'Completo' o tanto 'Mañana' como 'Tarde'
        if "Completo" in horarios_empleado:
            return True
        if "Mañana" in horarios_empleado and "Tarde" in horarios_empleado:
            return True
        return False

    # Caso 2: El turno es 'Mañana' o 'Tarde'
    if horario_turno in horarios_empleado:
        return True
    if "Completo" in horarios_empleado:
        return True

    return False


def puede_trabajar_mas(empleado_id, asignaciones_actuales, empleados, horas_turno=5):
    """
    Verifica que el empleado no exceda sus horas_max_semana si se le asigna
    un turno adicional de `horas_turno` horas.
    """
    # Buscar el empleado
    empleado = None
    for e in empleados:
        if e["id"] == empleado_id:
            empleado = e
            break

    if empleado is None:
        return False

    # Contar horas ya asignadas
    horas_actuales = 0
    for turno_id, lista_empleados in asignaciones_actuales.items():
        if empleado_id in lista_empleados:
            horas_actuales += horas_turno

    # Verificar si puede tomar más
    return (horas_actuales + horas_turno) <= empleado["horas_max_semana"]


def ya_trabaja_ese_dia(empleado_id, dia, asignaciones_actuales, turnos):
    """
    Verifica que el empleado no tenga ya un turno asignado en ese mismo día.
    Restricción: máximo 1 turno por día por empleado.
    """
    # Buscar turnos del mismo día
    for turno in turnos:
        if turno["dia"] == dia:
            turno_id = turno["id"]
            if turno_id in asignaciones_actuales:
                if empleado_id in asignaciones_actuales[turno_id]:
                    return True
    return False


def validar_asignacion_completa(asignaciones, turnos, empleados):
    """
    Valida la asignación completa y retorna un reporte detallado.
    
    Args:
        asignaciones: dict {turno_id: [empleado_id, ...]}
        turnos: lista de dicts de turnos
        empleados: lista de dicts de empleados
    
    Returns:
        dict con métricas de validación
    """
    horas_por_empleado = {}
    for e in empleados:
        horas_por_empleado[e["id"]] = 0

    turnos_cubiertos = 0
    turnos_sin_cubrir = []
    violaciones_horas = []
    violaciones_disponibilidad = []
    violaciones_dia_duplicado = []

    # Mapa de empleados para búsqueda rápida
    mapa_empleados = {e["id"]: e for e in empleados}
    mapa_turnos = {t["id"]: t for t in turnos}

    for turno in turnos:
        turno_id = turno["id"]
        asignados = asignaciones.get(turno_id, [])

        if len(asignados) >= turno["empleados_requeridos"]:
            turnos_cubiertos += 1
        else:
            turnos_sin_cubrir.append({
                "turno": turno_id,
                "requeridos": turno["empleados_requeridos"],
                "asignados": len(asignados)
            })

        for emp_id in asignados:
            # Sumar horas
            horas_por_empleado[emp_id] = horas_por_empleado.get(emp_id, 0) + turno["horas"]

            # Verificar disponibilidad
            emp = mapa_empleados.get(emp_id)
            if emp and not esta_disponible(emp, turno):
                violaciones_disponibilidad.append({
                    "empleado": emp_id,
                    "turno": turno_id,
                    "motivo": f"{emp['nombre']} no disponible {turno['dia']} {turno['horario']}"
                })

    # Verificar horas máximas
    for emp in empleados:
        horas = horas_por_empleado.get(emp["id"], 0)
        if horas > emp["horas_max_semana"]:
            violaciones_horas.append({
                "empleado": emp["id"],
                "nombre": emp["nombre"],
                "horas_asignadas": horas,
                "horas_max": emp["horas_max_semana"]
            })

    # Verificar duplicados por día
    for turno in turnos:
        asignados = asignaciones.get(turno["id"], [])
        for emp_id in asignados:
            # Contar cuántos turnos tiene este empleado en este día
            turnos_mismo_dia = 0
            for t2 in turnos:
                if t2["dia"] == turno["dia"] and emp_id in asignaciones.get(t2["id"], []):
                    turnos_mismo_dia += 1
            if turnos_mismo_dia > 1:
                violacion = {
                    "empleado": emp_id,
                    "dia": turno["dia"],
                    "turnos_asignados": turnos_mismo_dia
                }
                if violacion not in violaciones_dia_duplicado:
                    violaciones_dia_duplicado.append(violacion)

    total_violaciones = (
        len(violaciones_horas) +
        len(violaciones_disponibilidad) +
        len(violaciones_dia_duplicado)
    )

    return {
        "turnos_cubiertos": turnos_cubiertos,
        "turnos_total": len(turnos),
        "turnos_sin_cubrir": turnos_sin_cubrir,
        "violaciones_horas": violaciones_horas,
        "violaciones_disponibilidad": violaciones_disponibilidad,
        "violaciones_dia_duplicado": violaciones_dia_duplicado,
        "total_violaciones": total_violaciones,
        "horas_por_empleado": horas_por_empleado,
        "es_valida": total_violaciones == 0 and len(turnos_sin_cubrir) == 0
    }
