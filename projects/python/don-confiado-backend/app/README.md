# Don Confiado – Backend (FastAPI)

Servicio backend con endpoints de chat, RAG (productos/proveedores) y utilidades de negocio.

## Requisitos
- Python 3.11+
- Postgres 14+ con extensión `pgvector`
- Clave Google Generative AI (`GOOGLE_API_KEY`)
- (Opcional) Supabase (`SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`)

## Instalación

1) Ubícate en esta carpeta
```bash
cd projects/python/don-confiado-backend/app
```

2) Crea y activa un entorno virtual
```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows (Git Bash): source .venv/Scripts/activate
```

3) Instala dependencias
```bash
pip install -r requirements.txt
```

4) Variables de entorno (.env)
```bash
# Google Gemini
GOOGLE_API_KEY=tu_api_key_de_google

# Postgres (con pgvector instalado)
donconfiado_db_user=usuario
donconfiado_db_password=clave
donconfiado_db_host=localhost
donconfiado_db_port=5432
donconfiado_db_dbname=donconfiado

# Opcional: Supabase (para chat v1.1)
SUPABASE_URL=tu_url_de_supabase
SUPABASE_SERVICE_ROLE_KEY=tu_service_role_key
```

## Ejecutar el servidor
```bash
python tribu-main.py
# http://127.0.0.1:8000
```

Swagger: `http://127.0.0.1:8000/docs`

## API Endpoints (con ejemplos)

### 1) Health (opcional)
```bash
curl -X GET http://127.0.0.1:8000/hello
```

### 2) Chat con memoria (v1.0)
```bash
curl -X POST http://127.0.0.1:8000/api/chat_v1.0 \
  -H "Content-Type: application/json" \
  -d '{"message":"Hola, ¿quién eres?","user_id":"usuario-demo"}'
```

### 3) Clasificación + extracción + Supabase (v1.1)
```bash
curl -X POST http://127.0.0.1:8000/api/chat_v1.1 \
  -H "Content-Type: application/json" \
  -d '{"message":"Quiero crear un proveedor NIT 900123456, Razón Social ACME","user_id":"usuario-demo"}'
```

### 4) Chat multimodal + extracción de facturas (v2.0)
```bash
curl -X POST http://127.0.0.1:8000/api/chat_v2.0 \
  -H "Content-Type: application/json" \
  -d '{"message":"Analiza esta factura y crea el producto y proveedor si aplica","user_id":"usuario-demo"}'
```

### 5) Inicializar pgvector e índices (RAG)
```bash
curl -X POST http://127.0.0.1:8000/api/setup_pgvector
```

### 6) Sincronizar embeddings (productos y proveedores)
```bash
curl -X POST http://127.0.0.1:8000/api/sync_embeddings
```

### 7) Chat RAG (clase 03)
```bash
curl -X POST http://127.0.0.1:8000/api/chat_clase_03 \
  -H "Content-Type: application/json" \
  -d '{"user_id":"usuario-demo","message":"¿Qué precio y stock tiene el SKU ABC-123?"}'
```

Notas RAG:
- Embeddings: `models/text-embedding-004` (768-dim) con `pgvector` y `vector_l2_ops`.
- Producto incluye `proveedor_nombre` en el contenido y en `metadata`.
- En esta versión se indexan productos y proveedores (clientes deshabilitado).

## Solución de problemas
- Falta `GOOGLE_API_KEY`: defínelo en `.env` (o el backend lo pedirá por consola la primera vez).
- `pgvector` no instalado: instala la extensión y corre `/api/setup_pgvector`.
- Error 400 en embeddings: confirma modelo `models/text-embedding-004` y vigencia de la API key.
