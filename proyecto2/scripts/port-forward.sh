#!/usr/bin/env bash
# Expone las interfaces web del proyecto con kubectl port-forward.
#  - En la VM (cluster "local" de Rancher): los NodePort quedan dentro del contenedor de
#    Rancher, asi que se usa port-forward con --address 0.0.0.0 (como en la guia del curso).
#  - En el Mac (Docker Desktop): igual, y se abren en http://localhost:<puerto>.
#
# Uso:  ./proyecto2/scripts/port-forward.sh                  (Ctrl+C detiene todo)
#       MLFLOW_PORT=5001 ./proyecto2/scripts/port-forward.sh  (cambiar un puerto local)
#
# En macOS el puerto 5000 lo ocupa "Receptor AirPlay": por eso alli MLflow usa 5001.
# En la VM, si accedes desde otro equipo, abre los puertos:
#   sudo firewall-cmd --add-port={5000,9001}/tcp
set -euo pipefail
NAMESPACE=mlops

if [[ "$(uname)" == "Darwin" ]]; then
  DEFAULT_MLFLOW_PORT=5001
  HOST_SHOWN=localhost
else
  DEFAULT_MLFLOW_PORT=5000
  HOST_SHOWN="$(hostname -I 2>/dev/null | awk '{print $1}')"
  HOST_SHOWN="${HOST_SHOWN:-localhost}"
fi
ADDRESS="${ADDRESS:-0.0.0.0}"

# servicio:puerto_local:puerto_servicio  (se iran agregando en cada fase)
FORWARDS=(
  "mlflow:${MLFLOW_PORT:-$DEFAULT_MLFLOW_PORT}:5000"
  "minio:${MINIO_CONSOLE_PORT:-9001}:9001"
)

pids=()
cleanup() {
  echo; echo "Deteniendo port-forwards..."
  # ${#pids[@]} es seguro con set -u incluso con el bash 3.2 de macOS (array vacio)
  if [[ ${#pids[@]} -gt 0 ]]; then kill "${pids[@]}" 2>/dev/null || true; fi
}
trap cleanup EXIT INT TERM

for f in "${FORWARDS[@]}"; do
  IFS=: read -r svc lport rport <<<"$f"
  kubectl port-forward --address "$ADDRESS" -n "$NAMESPACE" "svc/$svc" "$lport:$rport" >/dev/null &
  pids+=($!)
  echo "  $svc  ->  http://$HOST_SHOWN:$lport"
done
wait
