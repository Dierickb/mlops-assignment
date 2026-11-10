#!/usr/bin/env bash
# Crea (o actualiza) los Secrets de Kubernetes a partir de proyecto2/.env.
# Asi las credenciales nunca se escriben en un manifiesto versionado en git.
#
# Uso (desde cualquier carpeta):   ./proyecto2/scripts/create-secrets.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="$ROOT/.env"
NAMESPACE=mlops

if [[ ! -f "$ENV_FILE" ]]; then
  echo "No existe $ENV_FILE. Copia .env.example a .env y cambia los valores." >&2
  exit 1
fi
set -a; source "$ENV_FILE"; set +a

kubectl apply -f "$ROOT/k8s/namespace.yaml"

# --dry-run=client -o yaml | kubectl apply: crea el Secret si no existe y lo actualiza si
# ya existe (un "kubectl create secret" a secas fallaria la segunda vez).
apply_secret() {
  local name="$1"; shift
  kubectl create secret generic "$name" -n "$NAMESPACE" "$@" \
    --dry-run=client -o yaml | kubectl apply -f -
}

apply_secret postgres-data-credentials \
  --from-literal=DATA_DB_USER="$DATA_DB_USER" \
  --from-literal=DATA_DB_PASSWORD="$DATA_DB_PASSWORD" \
  --from-literal=DATA_DB_NAME="$DATA_DB_NAME"

apply_secret postgres-mlflow-credentials \
  --from-literal=MLFLOW_DB_USER="$MLFLOW_DB_USER" \
  --from-literal=MLFLOW_DB_PASSWORD="$MLFLOW_DB_PASSWORD" \
  --from-literal=MLFLOW_DB_NAME="$MLFLOW_DB_NAME"

apply_secret minio-credentials \
  --from-literal=MINIO_ROOT_USER="$MINIO_ROOT_USER" \
  --from-literal=MINIO_ROOT_PASSWORD="$MINIO_ROOT_PASSWORD"

echo "Secrets listos en el namespace $NAMESPACE:"
kubectl get secrets -n "$NAMESPACE"
