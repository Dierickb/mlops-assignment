# Proyecto 2 — MLOps en Kubernetes (Nivel 3)

**Curso:** Operaciones de Machine Learning, Pontificia Universidad Javeriana · **Grupo:** 7

Arquitectura de MLOps completa desplegada en Kubernetes sobre el dataset *Diabetes 130-US hospitals*. Airflow carga los datos por lotes (raw → clean), entrena y registra modelos en MLflow y promueve el mejor con el alias `champion`. Una API FastAPI sirve ese modelo y registra cada inferencia, Streamlit la consume, Locust la somete a carga y Prometheus/Grafana la observan.

> **Estado:** en construcción por fases. Este README se completará con la arquitectura, las decisiones y las instrucciones de validación al final del proyecto.

## Fases

| Fase | Contenido | Estado |
|---|---|---|
| 1 | Namespace, ConfigMap, Secrets, Postgres de datos, Postgres de MLflow, MinIO, servidor MLflow | ⏳ manifiestos listos, falta desplegar |
| 2 | Airflow (Helm chart oficial) + DAG: lotes → raw → validación → clean → split → entrenamiento → MLflow → promoción | pendiente |
| 3 | API FastAPI (modelo `champion` desde MLflow, registro de inferencias, `/metrics`) + Streamlit | pendiente |
| 4 | Prometheus + Grafana + Locust, análisis del límite operativo de la API | pendiente |
| 5 | Imágenes en DockerHub, README final, video | pendiente |

## Decisiones tomadas hasta ahora

| Tema | Decisión | Por qué |
|---|---|---|
| Clúster | Clúster `local` de Rancher, el K3s dentro del contenedor, en la VM del taller | Es lo que plantea la guía del curso para una sola VM. La guía advierte que no se debe registrar como nodo de otro clúster la misma VM donde corre Rancher. |
| Airflow | Helm chart oficial **1.20.0**, que corresponde a **Airflow 3.1.8**, con LocalExecutor y los DAGs dentro de la imagen | Sigue la guía del curso (`Niveles/3/Airflow`). Fijar la versión del chart evita que su `appVersion` cambie y deje de coincidir con la imagen. |
| Versiones de ML | Python 3.12 con scikit-learn 1.8.0, numpy 2.4.2, pandas 2.3.3, scipy 1.17.1, joblib 1.5.3 y mlflow-skinny 3.16.1 | Son las versiones que fija el archivo de constraints de Airflow 3.1.8. Verificamos que mlflow-skinny 3.16.1 resuelve sin conflictos con ellas. La API usará **exactamente** las mismas: es la lección del Proyecto 1, donde el modelo dependía de las versiones del entorno de entrenamiento. |
| MLflow | Servidor 3.16.1 con `--serve-artifacts`: metadata en Postgres y artefactos en MinIO | SQLite no se acepta. Con este modo, Airflow y la API solo necesitan `MLFLOW_TRACKING_URI`, sin credenciales de S3 (mismo diseño que taller004). |
| Bases de datos | Tres instancias de Postgres: metadata de Airflow (la del chart), datos (`raw`, `clean`, `inference`) y metadata de MLflow | Separa el ciclo de vida de cada herramienta del de los datos de negocio. |
| Configuración | Kustomize (`kubectl apply -k k8s/`). Secrets generados desde `.env` por un script | Un solo comando despliega todo. El usuario de DockerHub y los tags se definen en un solo lugar. Las credenciales no quedan en git. |
| Acceso | `kubectl port-forward --address 0.0.0.0` (`scripts/port-forward.sh`) | En el clúster `local` los NodePort quedan dentro del contenedor de Rancher. Es el mismo método de la guía del curso. |
| Imágenes | Se construyen con `docker buildx --platform linux/amd64` y se publican en DockerHub | El Mac es ARM y la VM es x86: una imagen ARM no arranca en el clúster. |

## Fase 1: despliegue

**Requisitos en la VM:** `kubectl` con el kubeconfig del clúster `local` (ver la guía `Niveles/3/Kubernetes`) y `helm`. Además, el clúster debe tener una StorageClass por defecto, porque los PVC la necesitan:

```bash
kubectl get nodes
kubectl get storageclass        # debe haber una marcada "(default)"
```

**1. Construir y publicar la imagen de MLflow** (desde el Mac, en la raíz del repo):

```bash
docker login
docker buildx build --platform linux/amd64 \
  -t docker.io/dierick/mlops-p2-mlflow:3.16.1 --push proyecto2/images/mlflow
```

El usuario de DockerHub (`dierick`) ya está en `k8s/kustomization.yaml`.

**2. Desplegar** (en la VM, después de `git pull`):

```bash
cd proyecto2
cp .env.example .env && vi .env      # cambiar las contraseñas (solo letras y números)
./scripts/create-secrets.sh
kubectl apply -k k8s/
kubectl get pods -n mlops -w         # esperar Running / Completed (Ctrl+C para salir)
```

**3. Verificar:**

```bash
./scripts/verify-fase1.sh                           # chequeo automatico + prueba de punta a punta
kubectl get pvc -n mlops                            # los 3 PVC deben quedar Bound
kubectl logs -n mlops job/minio-create-bucket       # "Bucket mlflow listo."
kubectl exec -n mlops postgres-data-0 -- psql -U mlops -d mlops_data -c '\dn'   # raw, clean, inference
./scripts/port-forward.sh                           # MLflow :5000 y consola MinIO :9001
```

## Estructura

```
proyecto2/
├── k8s/                     manifiestos (kustomize)
│   ├── kustomization.yaml   punto de entrada; usuario de DockerHub y tags de imágenes
│   ├── namespace.yaml
│   ├── config/              ConfigMap compartido
│   ├── postgres-data/       StatefulSet + script SQL de esquemas (raw/clean/inference)
│   ├── postgres-mlflow/     StatefulSet del backend store de MLflow
│   ├── minio/               StatefulSet + Job que crea el bucket
│   └── mlflow/              Deployment del servidor de tracking
├── images/mlflow/           Dockerfile + pyproject + uv.lock del servidor MLflow
├── scripts/                 create-secrets.sh, port-forward.sh, verify-fase1.sh
└── .env.example             plantilla de credenciales (.env no se versiona)
```
