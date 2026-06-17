# 🧠 Features del MLP vs Heurística — Proyecto Scheduling con IA

## 📌 Resumen

Este documento describe las **características (features)** que utiliza cada método de asignación de turnos: la **Heurística Greedy** y la **Red Neuronal MLP (Multi-Layer Perceptron)**.

---

## ⚙️ Heurística Greedy — Criterios de Decisión

La heurística **NO usa features numéricas**. En su lugar, aplica **reglas determinísticas** en orden de prioridad:

| # | Criterio | Descripción | Tipo |
|---|----------|-------------|------|
| 1 | **Disponibilidad** | ¿El empleado trabaja ese día y horario? | Filtro binario (sí/no) |
| 2 | **Horas restantes** | ¿Puede trabajar 5h más sin exceder su máximo semanal? | Filtro binario (sí/no) |
| 3 | **Un turno por día** | ¿Ya fue asignado a otro turno ese mismo día? | Filtro binario (sí/no) |
| 4 | **Menor carga** | Entre los candidatos válidos, elige al que tenga **menos horas acumuladas** | Ordenamiento |

### Flujo de la Heurística:
```
Para cada turno (en orden T1, T2, ..., T8):
  1. Filtrar empleados disponibles en ese día/horario
  2. Filtrar los que no exceden horas máximas
  3. Filtrar los que no trabajan ya ese día
  4. Ordenar por horas acumuladas (ascendente)
  5. Asignar los primeros N empleados (N = requeridos)
```

### Ventajas:
- ✅ Rápido (O(n·m) donde n=turnos, m=empleados)
- ✅ Determinístico (siempre da el mismo resultado)
- ✅ Fácil de implementar y entender

### Limitaciones:
- ❌ **Miope**: Solo ve el turno actual, no planifica a futuro
- ❌ No optimiza globalmente (puede dejar turnos sin cubrir al final)
- ❌ El orden de procesamiento de turnos afecta el resultado

---

## 🧠 Red Neuronal MLP — Features de Entrada

El MLP recibe un **vector de 10 features numéricas** por cada par (empleado, turno) y predice la **probabilidad de asignación** (0.0 a 1.0).

### Vector de 10 Features:

| # | Feature | Fórmula | Rango | Descripción |
|---|---------|---------|-------|-------------|
| 0 | `horas_max_norm` | `horas_max_semana / 40` | [0, 1.2] | Capacidad máxima del empleado normalizada |
| 1 | `horas_asignadas` | `horas_acumuladas / 40` | [0, 1.2] | Cuántas horas ya lleva asignadas |
| 2 | `horas_restantes` | `max(0, max - asignadas) / 40` | [0, 1.2] | Cuántas horas más puede trabajar |
| 3 | `disponible` | `esta_disponible(emp, turno)` | {0, 1} | ¿Puede trabajar ese día/horario? |
| 4 | `dia_norm` | `DIAS_MAP[dia] / 6` | [0, 1] | Día de la semana codificado (Lun=0, Dom=1) |
| 5 | `horario_norm` | `HORARIOS_MAP[horario] / 2` | [0, 0.5, 1] | Franja horaria (Mañana=0, Tarde=0.5, Completo=1) |
| 6 | `emp_requeridos` | `turno.requeridos / 3` | [0, 1] | Cuántos empleados necesita el turno |
| 7 | `ya_trabaja_dia` | `1 si ya asignado ese día` | {0, 1} | ¿Ya tiene un turno ese día? |
| 8 | `ratio_uso` | `asignadas / max` | [0, 1] | Porcentaje de capacidad utilizada |
| 9 | `dias_disponibles` | `len(disponibilidad) / 7` | [0, 1] | Flexibilidad del empleado |

### Arquitectura del MLP:

```
Input(10) → Dense(64, ReLU) → Dense(32, ReLU) → Dense(16, ReLU) → Dense(1, Sigmoid)
```

| Capa | Neuronas | Activación | Parámetros |
|------|----------|------------|------------|
| Entrada | 10 | — | — |
| Oculta 1 | 64 | ReLU | 704 (10×64 + 64 bias) |
| Oculta 2 | 32 | ReLU | 2,080 (64×32 + 32 bias) |
| Oculta 3 | 16 | ReLU | 528 (32×16 + 16 bias) |
| Salida | 1 | Sigmoid | 17 (16×1 + 1 bias) |
| **Total** | | | **3,329 parámetros** |

### Hiperparámetros:

| Parámetro | Valor | Justificación |
|-----------|-------|---------------|
| Épocas | 50 | Suficiente para convergencia sin sobreajuste |
| Batch Size | 32 | Balance entre velocidad y estabilidad de gradientes |
| Optimizador | Adam (lr=0.001) | Adapta tasa de aprendizaje por parámetro |
| Loss | Binary Crossentropy | Estándar para clasificación binaria |
| Train/Val Split | 80% / 20% | Validación para detectar sobreajuste |
| Dataset | 20,000 muestras | 500 soluciones × 5 emp × 8 turnos |

### Funciones de Activación:

| Función | Fórmula | Dónde se usa | Por qué |
|---------|---------|-------------|---------|
| **ReLU** | f(x) = max(0, x) | Capas ocultas | Rápida, evita desvanecimiento del gradiente |
| **Sigmoid** | f(x) = 1/(1+e⁻ˣ) | Capa de salida | Salida [0,1] = probabilidad de asignación |
| ~~Tanh~~ | f(x) = tanh(x) | No usada | Más lenta que ReLU, susceptible a vanishing gradient |

### Ventajas del MLP:
- ✅ Aprende patrones complejos no lineales
- ✅ Generaliza a combinaciones no vistas en entrenamiento
- ✅ Puede mejorar con más datos

### Limitaciones del MLP:
- ❌ Requiere dataset de entrenamiento (generado sintéticamente)
- ❌ Mayor costo computacional
- ❌ Puede sobreajustar con pocos datos

---

## 🏆 Comparación Directa

| Aspecto | Heurística | MLP |
|---------|-----------|-----|
| **Entradas** | Reglas fijas | 10 features numéricas |
| **Decisión** | Determinística, secuencial | Probabilística, paralela |
| **Aprendizaje** | No aprende | Aprende de datos |
| **Velocidad** | Instantánea | Requiere entrenamiento previo |
| **Adaptabilidad** | Rígida | Se adapta a nuevos patrones |
| **Optimalidad** | Local (greedy) | Puede acercarse a global |
| **Interpretabilidad** | Alta (reglas claras) | Baja (caja negra) |
| **Escalabilidad** | Buena | Excelente con más datos |

---

*Proyecto Integrador — IA Moderna Aplicada al Scheduling*  
*TensorFlow/Keras • Python • Flask*
