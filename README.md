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