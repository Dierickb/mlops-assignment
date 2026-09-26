# proyecto1/dev — entorno local para probar DAGs

No se despliega. Sirve para escribir y probar `../dags/*.py` en tu Mac antes de subirlos.
El contenedor real (Python 3.7, `../airflow/Dockerfile`) sigue siendo la fuente de verdad.

## Primera vez (o si el entorno se rompe)

```bash
cd proyecto1/dev
rm -rf .venv
uv sync
```

## Verificar los DAGs (lo que harias cada vez que cambias el DAG)

```bash
./check_dags.sh
```

Carga la carpeta `../dags` con `DagBag`, igual que el scheduler de Airflow, y muestra
errores de importacion o la lista de DAGs/tareas encontradas.

## Reglas para no romper el entorno

- Las versiones de Airflow 2.6.0 estan fijadas en `[tool.uv].constraint-dependencies`
  de `pyproject.toml`. Por eso `uv add`, `uv sync` y `uv run` ya NO necesitan `-c`.
- Para agregar una libreria: `uv add <paquete>` (si choca con Airflow, uv lo dira).
- Ejecuta `uv` solo desde tu Terminal de macOS.
