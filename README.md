# zLightGBM

zLightGBM es un fork experimental de LightGBM que se instala como un paquete separado para no interferir con la instalación original de LightGBM.

## Instalación

### Prerrequisitos

- Python 3.9 o superior
- Git
- Herramientas de compilación (compilador C++, CMake, etc.)

### Pasos de instalación

#### 1. Clonar el repositorio

```bash
git clone --recursive https://github.com/fcalabrow/zLightGBM.git
cd zLightGBM
```

#### 2. Activar un entorno virtual

**Opción A: Usando venv tradicional**

```bash
# Crear el entorno virtual si no existe
python -m venv .venv

# Activar el entorno virtual
source .venv/bin/activate
```

**Opción B: Usando uv**

```bash
# Crear y activar el entorno virtual con uv
uv venv
source .venv/bin/activate
```

#### 3. Instalar zlightgbm

Una vez que el entorno virtual esté activado, ejecuta el script de instalación:

```bash
sh ./build-python.sh install
```

Este comando compilará la biblioteca C++ y instalará el paquete Python `zlightgbm` en tu entorno virtual.

### Verificar la instalación

Para verificar que la instalación fue exitosa, puedes importar el paquete en Python:

```python
import zlightgbm as lgb

# Verificar la versión
print(lgb.__version__)
```

## Uso

Una vez instalado, puedes usar zlightgbm de la misma manera que usarías lightgbm, pero importándolo como `zlightgbm`:

```python
import zlightgbm as lgb
import numpy as np

# Crear datos de ejemplo
X_train = np.random.rand(100, 10)
y_train = np.random.rand(100)

# Crear dataset
train_data = lgb.Dataset(X_train, label=y_train)

# Entrenar modelo
params = {'objective': 'regression', 'metric': 'rmse'}
model = lgb.train(params, train_data, num_boost_round=10)
```

## Características especiales de zLightGBM

zLightGBM incluye características experimentales para el control de overfitting que no están disponibles en LightGBM estándar:

### Canaritos (Canaries)

Los **canaritos** son variables aleatorias que se agregan al principio del dataset para detectar overfitting. Si el modelo comienza a usar principalmente los canaritos para hacer splits, zLightGBM se detiene automáticamente.

**Uso de canaritos:**

```python
import zlightgbm as zlgb
import numpy as np
import pandas as pd

# Preparar datos
X_train = np.random.rand(1000, 50)
y_train = np.random.randint(0, 2, 1000)

# Crear canaritos (variables aleatorias)
n_canaritos = 5
canaritos = np.random.rand(1000, n_canaritos)

# IMPORTANTE: Los canaritos deben ir al PRINCIPIO del dataset
X_train_with_canaritos = np.hstack([canaritos, X_train])

# Convertir a DataFrame para mantener el orden
df_train = pd.DataFrame(X_train_with_canaritos)
df_train.columns = [f'canarito_{i+1}' for i in range(n_canaritos)] + \
                   [f'feature_{i+1}' for i in range(50)]

# Crear dataset de LightGBM
train_data = zlgb.Dataset(df_train, label=y_train)

# Parámetros con canaritos
params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'canaritos': n_canaritos,  # Número de columnas canaritos al principio
    'learning_rate': 1.0,  # Se recomienda 1.0 cuando se usa gradient_bound
    'gradient_bound': 0.01,  # Límite del gradiente
    'num_iterations': 9999,  # zLightGBM se detiene solo si detecta overfitting
    'num_leaves': 9999,  # zLightGBM sabe cuándo no hacer más splits
    'min_data_in_leaf': 2000,
    'feature_fraction': 0.5,
    'verbosity': -1
}

# Entrenar modelo
# zLightGBM se detendrá automáticamente si solo encuentra splits en canaritos
model = zlgb.train(params, train_data)
```

**Características importantes de los canaritos:**

- Los canaritos **deben estar al principio** del dataset (primeras columnas)
- El parámetro `canaritos` indica cuántas columnas canaritos hay al principio
- Si `canaritos = -1` (default), la funcionalidad está desactivada
- zLightGBM se detiene automáticamente si solo encuentra splits en canaritos, indicando overfitting

### Gradient Bound

El parámetro `gradient_bound` limita el valor absoluto máximo del gradiente. Cuando está activado (> 0), ajusta el `learning_rate` dinámicamente.

**Uso de gradient_bound:**

```python
params = {
    'objective': 'binary',
    'learning_rate': 1.0,  # Se deja en 1.0 para que gradient_bound controle el escalado
    'gradient_bound': 0.01,  # Límite del gradiente (default: 0.0 = desactivado)
    # ... otros parámetros
}
```

**Notas sobre gradient_bound:**

- Si `gradient_bound = 0.0` (default), no tiene efecto
- Cuando `gradient_bound > 0`, el `learning_rate` se ajusta automáticamente
- Se recomienda usar `learning_rate = 1.0` cuando se usa `gradient_bound`

### Detención automática

zLightGBM puede detenerse automáticamente en dos casos:

1. **Solo canaritos**: Si el mejor split disponible es en un canarito, el entrenamiento se detiene
2. **Ganancia negativa**: Si el mejor split tiene ganancia <= 0, el entrenamiento se detiene

Por esta razón, puedes dejar `num_iterations` y `num_leaves` en valores altos (ej: 9999) y zLightGBM se detendrá cuando sea apropiado.

## Notas importantes

- Este paquete se instala como `zlightgbm` y no interfiere con instalaciones existentes de `lightgbm`
- Puedes tener ambos paquetes instalados simultáneamente en el mismo entorno
- Para usar este fork, siempre importa como `import zlightgbm` en lugar de `import lightgbm`
- Los canaritos son una característica experimental para detectar overfitting

## Repositorio

- URL: https://github.com/fcalabrow/zLightGBM
