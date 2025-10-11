# don-confiado

Plataforma open-source para asistentes empresariales y automatizaciones. El repo incluye:

- Backend en Python con FastAPI para endpoints de chat y negocio
- Servicio TypeScript para manejo de WhatsApp QR

## Requisitos

- Python 3.11+
- Node.js 18+
- Yarn 1.x o 3.x
- Cuenta/clave de Google Generative AI (Gemini): `GOOGLE_API_KEY`
- (Opcional) Supabase: `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`

## Estructura principal

- `projects/python/don-confiado-backend/app/`: backend FastAPI
- `projects/typescript/don-confiado-whatsapp-qr/`: servicio de WhatsApp QR

## Backend (FastAPI)

1) Crear entorno y dependencias

```bash
cd projects/python/don-confiado-backend/app
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

2) Variables de entorno

Crea un archivo `.env` en `projects/python/don-confiado-backend/app` con:

```bash
GOOGLE_API_KEY=tu_api_key_de_google
# Opcional, requerido para registrar distribuidores
SUPABASE_URL=tu_url_de_supabase
SUPABASE_SERVICE_ROLE_KEY=tu_service_role_key
```

3) Ejecutar el servidor

```bash
cd projects/python/don-confiado-backend/app
python tribu-main.py
# El servidor inicia en http://127.0.0.1:8000
```

4) Probar endpoints (ejemplos)

- Documentación interactiva (Swagger): `http://127.0.0.1:8000/docs`

Chat con memoria en sesión:

```bash
curl -X POST http://127.0.0.1:8000/api/chat_v1.0 \
  -H 'Content-Type: application/json' \
  -d '{"message":"Hola, ¿quién eres?","user_id":"usuario-demo"}'
```

Clasificación + extracción + registro en Supabase:

```bash
curl -X POST http://127.0.0.1:8000/api/chat_v1.1 \
  -H 'Content-Type: application/json' \
  -d '{"message":"Quiero crear un proveedor NIT 900123456, Razón Social ACME","user_id":"usuario-demo"}'
```

Notas:
- Si no defines `GOOGLE_API_KEY`, el backend te lo pedirá por consola la primera vez.
- Para crear distribuidores en Supabase, debes definir `SUPABASE_URL` y `SUPABASE_SERVICE_ROLE_KEY`.

## Servicio WhatsApp QR (TypeScript)

1) Instalar dependencias y ejecutar

```bash
cd projects/typescript/don-confiado-whatsapp-qr
yarn
yarn tsx src/index.ts
```

Sigue las instrucciones en la consola para escanear el QR. Revisa también el `README.md` dentro de ese proyecto para detalles específicos.

## Desarrollo

- Backend: para reiniciar, para el servidor y ejecuta de nuevo `python tribu-main.py`.
- Código formateado y tipado: respeta las versiones listadas en `requirements.txt` (backend) y usa Node 18+ (TS).

## Solución de problemas

- Error por credenciales de Supabase: agrega `SUPABASE_URL` y `SUPABASE_SERVICE_ROLE_KEY` al `.env` del backend.
- Error por `GOOGLE_API_KEY`: crea el `.env` con esa clave o proporciónala cuando el backend la solicite.
- Puerto ocupado (8000): cierra el proceso previo o cambia el puerto en `tribu-main.py`.

## SQL: Tabla `terceros` (Supabase/Postgres)

Ejecuta esta consulta en tu base de datos (por ejemplo, en el SQL Editor de Supabase) para crear la tabla `public.terceros` que usa el endpoint `chat_v1.1`.

```sql
create table public.terceros (
  id serial not null,
  tipo_documento character varying(5) not null,
  numero_documento character varying(30) not null,
  razon_social character varying(200) null,
  nombres character varying(100) null,
  apellidos character varying(100) null,
  telefono_fijo character varying(20) null,
  telefono_celular character varying(20) null,
  tipo_tercero character varying(20) not null,
  direccion text null,
  email character varying(150) null,
  email_facturacion character varying(150) null,
  fecha_creacion timestamp without time zone null default CURRENT_TIMESTAMP,
  constraint terceros_pkey primary key (id),
  constraint uq_documento unique (tipo_documento, numero_documento),
  constraint terceros_tipo_documento_check check (
    (
      (tipo_documento)::text = any (
        (
          array[
            'CC'::character varying,
            'NIT'::character varying,
            'CE'::character varying
          ]
        )::text[]
      )
    )
  ),
  constraint terceros_tipo_tercero_check check (
    (
      (tipo_tercero)::text = any (
        (
          array[
            'cliente'::character varying,
            'proveedor'::character varying,
            'empleado'::character varying
          ]
        )::text[]
      )
    )
  )
) TABLESPACE pg_default;
```

```sql
create table public.productos (
  id serial not null,
  sku character varying(50) not null,
  nombre character varying(200) not null,
  precio_venta numeric(15, 2) not null,
  cantidad integer not null default 0,
  proveedor_id integer null,
  fecha_creacion timestamp without time zone null default CURRENT_TIMESTAMP,
  constraint productos_pkey primary key (id),
  constraint productos_sku_key unique (sku),
  constraint productos_proveedor_id_fkey foreign KEY (proveedor_id) references terceros (id),
  constraint productos_cantidad_check check ((cantidad >= 0)),
  constraint productos_precio_venta_check check ((precio_venta >= (0)::numeric))
) TABLESPACE pg_default;
```