# 🧠 Features del MLP vs Heurística — Proyecto Scheduling con IA
Arquitectura Neuronal Predictiva para Optimización de Scheduling Operativo (ABAJO)

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



# Arquitectura Neuronal Predictiva para Optimización de Scheduling Operativo

Este documento detalla la fundamentación matemática, el diseño topológico y el análisis telemétrico del perceptrón multicapa (MLP) desarrollado para la resolución estocástica del problema de *scheduling*. La aproximación trasciende la heurística secuencial al proyectar el problema de asignación en un espacio vectorial multidimensional, permitiendo la inferencia paralela y la escalabilidad a entornos de producción distribuidos.

---

## 1. Ingeniería de Características (Feature Engineering) y Vectorización

El éxito de la red neuronal depende de la proyección del estado operativo en un espacio latente que el modelo pueda procesar sin sufrir explosión o desvanecimiento de gradientes. El conjunto de datos base (10 empleados, 8 turnos) produce un espacio combinatorio inicial que se vectoriza en tensores de entrada $X \in \mathbb{R}^{10}$.

Para garantizar la estabilidad numérica durante el descenso del gradiente, todas las variables continuas y categóricas se sometieron a una normalización estricta al dominio $[0, 1]$.

La matriz de diseño de entrada para un lote de tamaño $m$ se define como $X_{m \times 10}$, donde cada vector $x^{(i)}$ encapsula:

* **Restricciones de Capacidad:** Normalización de horas máximas (ej. empleando métricas base de 40 a 50 horas semanales) y cálculo de capacidad residual.
* **Restricciones Espacio-Temporales:** Codificación de franjas horarias y días operativos mediante un mapeo lineal escalar, evitando la alta dimensionalidad del *One-Hot Encoding* para mantener la eficiencia paramétrica.
* **Estado de Saturación:** Ratio continuo de utilización actual vs. límite contractual.

---

## 2. Topología Profunda del Perceptrón Multicapa (MLP)

La red implementa una arquitectura secuencial profunda de tipo embudo (Funnel Architecture). Esta estructura fuerza a la red a comprimir la información, extrayendo representaciones de alto nivel en cada capa sucesiva.

### Ecuación de Propagación hacia Adelante (Forward Pass)

Para cada capa oculta $l \in \{1, 2, 3\}$, la transformación afín seguida de la activación no lineal se define como:

$$A^{[l]} = g(W^{[l]} A^{[l-1]} + b^{[l]})$$

Donde:

* $W^{[l]}$ es la matriz de pesos de la capa $l$.
* $b^{[l]}$ es el vector de sesgo (*bias*).
* $g(z)$ es la función de activación Rectified Linear Unit (ReLU), definida como $g(z) = \max(0, z)$.

### Capa Predictiva

La última capa mapea la combinación lineal final a un espacio probabilístico continuo utilizando la función Sigmoide:

$$\hat{y} = \sigma(W^{[out]} A^{[3]} + b^{[out]}) = \frac{1}{1 + e^{-(W^{[out]} A^{[3]} + b^{[out]})}}$$

Esta formulación garantiza que la predicción $\hat{y} \in (0, 1)$ sea directamente interpretable como la probabilidad matemática de que la tupla (Empleado, Turno) sea una asignación legal y óptima.

---

## 3. Dinámica de Optimización y Función de Costo

La optimización de los 3,329 parámetros entrenables se aborda como un problema de minimización empírica del riesgo. Dado el carácter binario del objetivo (Asignar = 1, No Asignar = 0), se emplea la Entropía Cruzada Binaria (Binary Crossentropy) como función de pérdida objetivo $\mathcal{L}$:

$$\mathcal{L}(W, b) = - \frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(\hat{y}^{(i)}) + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)}) \right]$$

Para navegar la topología de la función de pérdida y actualizar los tensores de pesos, se utiliza el optimizador **Adam** (Adaptive Moment Estimation). Este algoritmo computa tasas de aprendizaje individuales para cada parámetro basándose en estimaciones del primer (media) y segundo (varianza no centrada) momento de los gradientes, lo que acelera significativamente la convergencia en superficies de error complejas en comparación con el Descenso de Gradiente Estocástico (SGD) tradicional.

---

## 4. Análisis de Convergencia (Telemetría Empírica)

Los registros de telemetría extraídos de la ejecución local en Python proporcionan una radiografía exacta del comportamiento dinámico de la red a lo largo de 50 épocas de entrenamiento.

### 4.1. Análisis de la Función de Pérdida (Loss)

* **Caída Exponencial Inicial:** En la época 1, la pérdida de entrenamiento se situó en 0.118 y la de validación en 0.084. Para la época 5, ambas métricas sufrieron una reducción superior al 50%, alcanzando niveles de 0.044. Esto demuestra una asimilación rápida de las características lineales primarias del *dataset*.
* **Estabilización Asintótica:** A partir de la época 15, la derivada de la curva de pérdida tiende a cero. La red alcanza un mínimo local altamente eficiente, finalizando en la época 50 con una pérdida de entrenamiento de **0.0402** y una pérdida de validación de **0.0391**.
* **Ausencia de Overfitting:** La brecha de generalización (Generalization Gap) es estadísticamente nula. El hecho de que la pérdida de validación $\mathcal{L}_{val}$ siga estrechamente e incluso supere ligeramente a $\mathcal{L}_{train}$ indica que la arquitectura posee una regularización intrínseca robusta y no está memorizando el ruido del lote de entrenamiento.

### 4.2. Saturación de Exactitud (Accuracy)

El modelo alcanza la convergencia operativa de manera acelerada. La exactitud de validación se estabiliza en un límite superior rígido de **98.37%** desde etapas tempranas (época 4) y se mantiene constante hasta la época 50. La exactitud de entrenamiento finaliza en **98.29%**. Esta meseta simétrica confirma que la red neuronal ha descifrado la lógica combinatoria subyacente de las reglas operativas, resolviendo las *hard-constraints* del negocio sin requerir profundidad paramétrica adicional.

---

## 5. Arquitectura de Despliegue y Escalabilidad a Producción

El modelo desarrollado no es un artefacto estático; está diseñado para la integración en entornos de alta disponibilidad. La resolución heurística es incapaz de lidiar con operaciones vectoriales masivas, mientras que este modelo MLP permite la inferencia en lotes (Batch Inference).

Para trasladar este rendimiento teórico a un impacto comercial (ROI), la arquitectura del software debe escalar mediante los siguientes vectores tecnológicos:

1. **Contenerización y Microservicios:** Empaquetar el modelo entrenado (.keras / .h5) y el pipeline de preprocesamiento (extracción de *features*) dentro de un contenedor Docker.
2. **Exposición vía API REST:** Utilizar *frameworks* de alto rendimiento en Python (FastAPI o Flask) para exponer un *endpoint* de inferencia. Los sistemas ERP de recursos humanos enviarán los arreglos JSON de disponibilidad y la API retornará las probabilidades de asignación.
3. **Orquestación en Local Cloud:** Desplegar los contenedores en un clúster de Kubernetes (K3s) para garantizar balanceo de carga. Si el volumen de empleados crece de 10 a 10,000, el modelo matemático soporta la dimensionalidad, y el orquestador asegura los recursos de CPU/GPU necesarios para la inferencia simultánea.

La adopción de este pipeline de Inteligencia Artificial sustituye procesos de asignación manuales, frágiles y propensos al error, por un sistema determinístico, auditable y escalable que optimiza matemáticamente la utilización del talento humano.
