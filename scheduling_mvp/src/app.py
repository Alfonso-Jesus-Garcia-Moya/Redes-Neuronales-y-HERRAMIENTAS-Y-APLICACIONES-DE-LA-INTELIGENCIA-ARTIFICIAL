"""
app.py - Fase 6: Visualización web con Flask.
Dos dashboards: estático (copia) y dinámico (AJAX en vivo).
"""
import json, os, sys
from flask import Flask, render_template, jsonify, request, redirect, url_for

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTADOS_DIR = os.path.join(BASE_DIR, "resultados")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
DATA_DIR = os.path.join(BASE_DIR, "data")
sys.path.insert(0, BASE_DIR)

app = Flask(__name__, template_folder=TEMPLATES_DIR)

def cargar_json(ruta):
    if os.path.exists(ruta):
        with open(ruta, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

def guardar_json(ruta, data):
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def _get_all_data():
    return {
        "empleados": cargar_json(os.path.join(DATA_DIR, "empleados.json")) or [],
        "turnos": cargar_json(os.path.join(DATA_DIR, "turnos.json")) or [],
        "resultado_heur": cargar_json(os.path.join(RESULTADOS_DIR, "asignacion_heuristica.json")),
        "resultado_mlp": cargar_json(os.path.join(RESULTADOS_DIR, "asignacion_mlp.json")),
        "comparacion": cargar_json(os.path.join(RESULTADOS_DIR, "comparacion.json")),
        "historial": cargar_json(os.path.join(RESULTADOS_DIR, "historial_entrenamiento.json")),
    }

# ─── DASHBOARD ESTÁTICO (copia original) ───
@app.route("/estatico")
def dashboard_estatico():
    d = _get_all_data()
    return render_template("dashboard_estatico.html", **d)

# ─── DASHBOARD DINÁMICO (principal) ───
@app.route("/")
def dashboard():
    d = _get_all_data()
    return render_template("dashboard_dinamico.html", **d)

# ─── API JSON: datos actuales ───
@app.route("/api/datos")
def api_datos():
    return jsonify(_get_all_data())

# ─── API: agregar empleado ───
@app.route("/api/agregar_empleado", methods=["POST"])
def agregar_empleado():
    empleados = cargar_json(os.path.join(DATA_DIR, "empleados.json")) or []
    data = request.get_json() if request.is_json else request.form
    nuevo_id = f"E{len(empleados)+1}"
    dias_sel = data.getlist("dias") if hasattr(data, "getlist") else data.get("dias", [])
    horario = data.get("horario", "Mañana")
    disponibilidad = {}
    for d in dias_sel:
        disponibilidad[d] = [horario] if horario != "Completo" else ["Mañana", "Tarde"]
    nuevo = {
        "id": nuevo_id,
        "nombre": data.get("nombre", "Nuevo"),
        "horas_max_semana": int(data.get("horas_max", 20)),
        "disponibilidad": disponibilidad
    }
    empleados.append(nuevo)
    guardar_json(os.path.join(DATA_DIR, "empleados.json"), empleados)
    if request.is_json:
        return jsonify({"ok": True, "empleado": nuevo, "total": len(empleados)})
    return redirect(url_for("dashboard"))

# ─── API: recalcular todo ───
@app.route("/api/recalcular", methods=["POST"])
def recalcular():
    from src.cargar_datos import cargar_empleados, cargar_turnos
    from src.heuristica import ejecutar_heuristica
    from src.generar_datos_entrenamiento import generar_dataset
    from src.modelo_mlp import entrenar_mlp, predecir_asignacion, guardar_historial
    from src.evaluacion import comparar_soluciones

    empleados = cargar_empleados()
    turnos = cargar_turnos()
    resultado_heur, _ = ejecutar_heuristica()
    X, y = generar_dataset(300)
    modelo, historial = entrenar_mlp(X, y, epochs=30, batch_size=32)
    guardar_historial(historial)
    resultado_mlp, _ = predecir_asignacion(modelo, empleados, turnos)
    comparar_soluciones(resultado_heur, resultado_mlp, empleados, turnos)

    if request.is_json:
        return jsonify({"ok": True, "data": _get_all_data()})
    return redirect(url_for("dashboard"))

def iniciar_app(host="127.0.0.1", port=5000):
    print(f"\n[FLASK] Dashboard dinámico: http://{host}:{port}")
    print(f"[FLASK] Dashboard estático: http://{host}:{port}/estatico")
    print(f"[FLASK] Ctrl+C para detener\n")
    app.run(host=host, port=port, debug=False)

if __name__ == "__main__":
    iniciar_app()
