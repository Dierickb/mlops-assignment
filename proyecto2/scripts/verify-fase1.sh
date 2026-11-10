#!/usr/bin/env bash
# Verificacion automatica de la Fase 1 (capa de datos + MLflow).
# Revisa cada pieza por separado y al final hace una prueba de punta a punta:
# registra un run con un artefacto en MLflow y confirma que el archivo llego a MinIO.
#
# Uso:  ./proyecto2/scripts/verify-fase1.sh
# No usa "set -e" a proposito: si un chequeo falla, sigue con los demas y al final resume.
set -uo pipefail
NS=mlops
pass=0; fail=0

ok() { echo "  ✅ $1"; pass=$((pass + 1)); }
ko() { echo "  ❌ $1"; fail=$((fail + 1)); }
check() {  # check "descripcion" comando...
  local desc="$1"; shift
  if "$@" >/dev/null 2>&1; then ok "$desc"; else ko "$desc"; fi
}
# Ejecuta SQL dentro del pod de Postgres indicado. El SQL entra por stdin para no pelear
# con comillas; usuario y base salen de las variables de entorno del propio pod.
sql() {  # sql <pod> "<consulta>"
  echo "$2" | kubectl exec -i -n "$NS" "$1" -- sh -c 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -tA' 2>/dev/null
}

echo "1. Pods y Job"
for app in postgres-data postgres-mlflow minio mlflow; do
  check "$app en estado Ready" kubectl wait -n "$NS" --for=condition=Ready pod -l "app=$app" --timeout=180s
done
check "Job minio-create-bucket completado" \
  kubectl wait -n "$NS" --for=condition=complete job/minio-create-bucket --timeout=180s

echo "2. Discos persistentes (PVC)"
for pvc in data-postgres-data-0 data-postgres-mlflow-0 data-minio-0; do
  phase="$(kubectl get pvc -n "$NS" "$pvc" -o jsonpath='{.status.phase}' 2>/dev/null)"
  [[ "$phase" == "Bound" ]] && ok "PVC $pvc Bound" || ko "PVC $pvc (estado: ${phase:-no existe})"
done

echo "3. Postgres de datos: esquemas y tablas del script de inicio"
n="$(sql postgres-data-0 "SELECT count(*) FROM information_schema.schemata WHERE schema_name IN ('raw','clean','inference');")"
[[ "$n" == "3" ]] && ok "esquemas raw, clean, inference" || ko "esquemas (encontrados: ${n:-error})"
n="$(sql postgres-data-0 "SELECT count(*) FROM information_schema.tables WHERE table_schema||'.'||table_name IN ('raw.load_batches','raw.diabetes','inference.predictions');")"
[[ "$n" == "3" ]] && ok "tablas raw.load_batches, raw.diabetes, inference.predictions" || ko "tablas (encontradas: ${n:-error})"

echo "4. Postgres de MLflow: MLflow creo sus tablas"
n="$(sql postgres-mlflow-0 "SELECT count(*) FROM information_schema.tables WHERE table_name IN ('experiments','runs','registered_models');")"
[[ "$n" == "3" ]] && ok "tablas experiments, runs, registered_models" || ko "tablas de MLflow (encontradas: ${n:-error})"

echo "5. MinIO: bucket de artefactos"
check "bucket 'mlflow' existe" kubectl exec -n "$NS" minio-0 -- sh -c \
  'mc alias set v http://localhost:9000 "$MINIO_ROOT_USER" "$MINIO_ROOT_PASSWORD" >/dev/null && mc ls v/mlflow'

echo "6. MLflow responde"
check "/health responde" kubectl exec -n "$NS" deploy/mlflow -- \
  python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/health')"

echo "7. Prueba de punta a punta: run + artefacto (cliente -> MLflow -> Postgres y MinIO)"
# MLflow imprime lineas informativas ("View experiment at ...") en stdout: el run_id es la ultima.
out="$(kubectl exec -i -n "$NS" deploy/mlflow -- python - 2>/dev/null <<'PY'
import os, tempfile, mlflow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("verificacion-fase1")
with mlflow.start_run(run_name="smoke-test") as run:
    mlflow.log_param("fase", 1)
    mlflow.log_metric("ok", 1.0)
    path = os.path.join(tempfile.mkdtemp(), "prueba.txt")
    with open(path, "w") as f:
        f.write("hola desde la fase 1")
    mlflow.log_artifact(path)          # viaja al servidor (--serve-artifacts) y de ahi a MinIO
# Descargarlo de vuelta prueba el camino inverso, el que usara la API para bajar modelos
local = mlflow.artifacts.download_artifacts(run_id=run.info.run_id, artifact_path="prueba.txt")
assert open(local).read() == "hola desde la fase 1"
print(run.info.run_id)
PY
)"
run_id="$(printf '%s\n' "$out" | tail -n 1 | tr -d '[:space:]')"
[[ "$run_id" =~ ^[0-9a-f]{32}$ ]] || run_id=""
if [[ -n "$run_id" ]]; then
  ok "run $run_id registrado y artefacto descargado de vuelta"
  n="$(sql postgres-mlflow-0 "SELECT count(*) FROM runs WHERE run_uuid='$run_id';")"
  [[ "$n" == "1" ]] && ok "el run quedo en postgres-mlflow (metadata)" || ko "run no encontrado en postgres-mlflow"
  # El grep corre en TU maquina, no dentro del pod: la imagen de MinIO es minima y no trae grep.
  found="$(kubectl exec -n "$NS" minio-0 -- sh -c \
    'mc alias set v http://localhost:9000 "$MINIO_ROOT_USER" "$MINIO_ROOT_PASSWORD" >/dev/null && mc find v/mlflow --name prueba.txt' \
    2>/dev/null | grep "$run_id")"
  [[ -n "$found" ]] && ok "prueba.txt quedo en MinIO: $found" || ko "prueba.txt no aparece en MinIO bajo el run $run_id"
else
  ko "no se pudo registrar el run (revisa: kubectl logs -n $NS deploy/mlflow)"
fi

echo
echo "Resultado: $pass OK, $fail con error"
[[ "$fail" -eq 0 ]]
