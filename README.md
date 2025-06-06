# Tarea 2 Computación en GPU

## Como Ejecutar los test

Primero que nada, hay que cargar el modulo googletest

```
git submodule update --init --recursive
```

Para buildear el proyecto y correr los test, hay que ejecutar

```
cmake -S . -B build -DBUILD_TESTING=ON
cmake --build build
cd build && ctest
```

Version CUDA
Requisitos:
- CMake 3.35+
- CUDA Toolkit 11.0+
- Compilador C++17
```
./build/test/benchmark
```
Experimentos Implementados:
- Tamaño de bloque: 
    - Múltiplos de 32 (bien alineados): 32, 64
    - No múltiplos (no alineados): 16, 17, 24

- Métodos de conteo:
    - Bucles anidados
    - Verificación explícita con ifs

Salida:
- results.csv (datos de performance para analisis)
- Columnas del CSV:
    -Implementation: Tipo de implementación (Sequential, CUDA)
    -GridSize: Tamaño del grid (128, 256, 512, 1024)
    -BlockSize: Tamaño del bloque CUDA (16, 17, 24, 32, 64) - 0 para Sequential
    -UseIfs: Método de conteo de vecinos (true=ifs explícitos, false=bucles)
    -Time Ms: Tiempo promedio de ejecución en milisegundos

Configuraciones Probadas:
    - Grids: 128x128, 256x256, 512x512, 1024x1024
    - Block sizes: 16, 17, 24, 32, 64
    - Métodos: bucle vs ifs
    - Iteraciones: 5 por configuración (promediado)