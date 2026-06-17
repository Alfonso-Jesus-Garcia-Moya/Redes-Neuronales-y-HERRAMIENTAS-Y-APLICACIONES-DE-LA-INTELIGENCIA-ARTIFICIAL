"""
modelo_mlp.py - Fase 4: Red Neuronal Multicapa (MLP) para predicción
de asignaciones de turnos.

Arquitectura del MLP:
  Input (10 features) → Dense(64, ReLU) → Dense(32, ReLU) → Dense(1, Sigmoid)

El modelo aprende a predecir la probabilidad de que un empleado
sea asignado a un turno dado, basándose en las features codificadas.
"""
import json
import os
import numpy as np

from src.cargar_datos import cargar_empleados, cargar_turnos
from src.restricciones import esta_disponible, puede_trabajar_mas, ya_trabaja_ese_dia
from src.generar_datos_entrenamiento import codificar_features, generar_dataset

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTADOS_DIR = os.path.join(BASE_DIR, "resultados")


def entrenar_mlp(X, y, epochs=50, batch_size=32):
    """
    Entrena un modelo MLP con Keras/TensorFlow.
    
    Args:
        X: array de features (n_muestras, 10)
        y: array de labels (n_muestras,)
        epochs: épocas de entrenamiento
        batch_size: tamaño del batch
    
    Returns:
        modelo entrenado, historial de entrenamiento
    """
    # Importar TensorFlow/Keras aquí para que no falle si no está instalado
    try:
        from tensorflow import keras
        from tensorflow.keras import layers
    except ImportError:
        from keras import layers
        import keras

    print("\n" + "=" * 60)
    print("  ENTRENANDO RED NEURONAL MLP")
    print("=" * 60)
    print(f"  Muestras: {X.shape[0]} | Features: {X.shape[1]}")
    print(f"  Epochs: {epochs} | Batch: {batch_size}")

    # Definir arquitectura del MLP
    modelo = keras.Sequential([
        layers.Input(shape=(X.shape[1],)),
        layers.Dense(64, activation='relu', name='capa_oculta_1'),
        layers.Dense(32, activation='relu', name='capa_oculta_2'),
        layers.Dense(16, activation='relu', name='capa_oculta_3'),
        layers.Dense(1, activation='sigmoid', name='salida')
    ])

    modelo.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    modelo.summary()

    # Dividir datos: 80% train, 20% validación
    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    # Entrenar
    historial = modelo.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )

    # Métricas finales
    loss_train, acc_train = modelo.evaluate(X_train, y_train, verbose=0)
    loss_val, acc_val = modelo.evaluate(X_val, y_val, verbose=0)
    print(f"\n  Accuracy Train: {acc_train:.4f} | Loss: {loss_train:.4f}")
    print(f"  Accuracy Val:   {acc_val:.4f} | Loss: {loss_val:.4f}")

    return modelo, historial


def predecir_asignacion(modelo, empleados=None, turnos=None):
    """
    Usa el MLP entrenado para generar una asignación de turnos.
    
    Estrategia:
      1. Para cada par (empleado, turno), predecir probabilidad
      2. Ordenar pares por probabilidad descendente
      3. Asignar greedily respetando restricciones
    
    Returns:
        dict resultado, dict asignaciones
    """
    if empleados is None:
        empleados = cargar_empleados()
    if turnos is None:
        turnos = cargar_turnos()

    print("\n" + "=" * 60)
    print("  PREDICCIÓN MLP - ASIGNACIÓN DE TURNOS")
    print("=" * 60)

    # Generar predicciones para todos los pares
    pares = []
    for turno in turnos:
        for emp in empleados:
            features = codificar_features(emp, turno, 0, False)
            pares.append({
                "empleado": emp,
                "turno": turno,
                "features": features
            })

    # Predecir probabilidades
    X_pred = np.array([p["features"] for p in pares], dtype=np.float32)
    probabilidades = modelo.predict(X_pred, verbose=0).flatten()

    for i, p in enumerate(pares):
        p["probabilidad"] = float(probabilidades[i])

    # Ordenar por probabilidad descendente
    pares.sort(key=lambda p: p["probabilidad"], reverse=True)

    # Asignar greedily respetando restricciones
    asignaciones = {}
    horas_asignadas = {emp["id"]: 0 for emp in empleados}

    for p in pares:
        emp = p["empleado"]
        turno = p["turno"]
        emp_id = emp["id"]
        turno_id = turno["id"]

        # ¿El turno ya tiene suficientes empleados?
        asignados_turno = asignaciones.get(turno_id, [])
        if len(asignados_turno) >= turno["empleados_requeridos"]:
            continue

        # Verificar restricciones
        if not esta_disponible(emp, turno):
            continue
        if not puede_trabajar_mas(emp_id, asignaciones, empleados, turno["horas"]):
            continue
        if ya_trabaja_ese_dia(emp_id, turno["dia"], asignaciones, turnos):
            continue

        # Asignar
        if turno_id not in asignaciones:
            asignaciones[turno_id] = []
        asignaciones[turno_id].append(emp_id)
        horas_asignadas[emp_id] += turno["horas"]

    # Log de asignaciones
    for turno in turnos:
        turno_id = turno["id"]
        asignados = asignaciones.get(turno_id, [])
        nombres = []
        for eid in asignados:
            emp = next(e for e in empleados if e["id"] == eid)
            nombres.append(emp["nombre"])
        requeridos = turno["empleados_requeridos"]
        estado = "✓" if len(asignados) >= requeridos else "✗ INCOMPLETO"
        print(f"  {turno_id} | {turno['dia']:10s} {turno['horario']:10s} | "
              f"Asignados: {nombres} | Prob max: {max((p['probabilidad'] for p in pares if p['turno']['id'] == turno_id), default=0):.3f} | {estado}")

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
        "metodo": "red_neuronal_mlp",
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

    # Guardar resultado
    os.makedirs(RESULTADOS_DIR, exist_ok=True)
    ruta_salida = os.path.join(RESULTADOS_DIR, "asignacion_mlp.json")
    with open(ruta_salida, "w", encoding="utf-8") as f:
        json.dump(resultado, f, ensure_ascii=False, indent=2)
    print(f"\n[GUARDADO] Resultado MLP → {ruta_salida}")

    # Resumen
    print("\n--- RESUMEN MLP ---")
    print(f"  Turnos cubiertos: {resultado['resumen']['turnos_cubiertos']}/{resultado['resumen']['turnos_total']}")
    for emp in empleados:
        h = horas_asignadas[emp["id"]]
        barra = "█" * (h // 5) + "░" * ((emp["horas_max_semana"] - h) // 5)
        print(f"    {emp['id']} {emp['nombre']:8s} | {h:2d}h / {emp['horas_max_semana']}h max | {barra}")

    return resultado, asignaciones


def guardar_historial(historial):
    """Guarda las métricas de entrenamiento para la visualización."""
    metricas = {
        "loss": [float(v) for v in historial.history["loss"]],
        "accuracy": [float(v) for v in historial.history["accuracy"]],
        "val_loss": [float(v) for v in historial.history["val_loss"]],
        "val_accuracy": [float(v) for v in historial.history["val_accuracy"]],
        "epochs": len(historial.history["loss"])
    }
    ruta = os.path.join(RESULTADOS_DIR, "historial_entrenamiento.json")
    os.makedirs(RESULTADOS_DIR, exist_ok=True)
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(metricas, f, ensure_ascii=False, indent=2)
    print(f"[GUARDADO] Historial entrenamiento → {ruta}")
    return metricas


if __name__ == "__main__":
    # Generar datos y entrenar
    X, y = generar_dataset(500)
    modelo, historial = entrenar_mlp(X, y, epochs=50)
    guardar_historial(historial)
    predecir_asignacion(modelo)
