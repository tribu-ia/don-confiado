
from typing import Optional, List, Literal, Any, Dict
from pydantic import BaseModel, Field

class PayloadCreateProvider(BaseModel):
    """Datos para crear un proveedor en el sistema."""
    nombre: Optional[str] = Field(None, description="Nombre o razón social del proveedor")
    nit: Optional[str] = Field(None, description="Número de identificación tributaria (NIT) del proveedor")
    direccion: Optional[str] = Field(None, description="Dirección física del proveedor")
    telefono: Optional[str] = Field(None, description="Número de teléfono de contacto del proveedor")

class PayloadCreateProduct(BaseModel):
    nombre: str = Field(..., description="Nombre del producto", max_length=200)
    precio_venta: float = Field(..., description="Precio de venta del producto", ge=0)
    cantidad: int = Field(0, description="Cantidad disponible del producto", ge=0)
    sku: Optional[str] = Field(None, description="Código SKU del producto (se genera automáticamente si no se proporciona)", max_length=50)
    proveedor: Optional[str] = Field(None, description="Nombre o NIT del proveedor del producto")

class PayloadCreateClient(BaseModel):
    """Datos para crear un cliente en el sistema."""
    nombre: str = Field(..., description="Nombre completo o razón social del cliente")
    nit: str = Field(..., description="Número de identificación tributaria (NIT) o documento de identidad del cliente")
    direccion: str = Field(..., description="Dirección física o de correspondencia del cliente")


class UserIntention(BaseModel):
    """
    Modelo de salida estructurada: intención del usuario + datos extraídos del mensaje o archivos adjuntos.
    Este modelo captura tanto la intención del usuario como los datos necesarios para ejecutar esa intención,
    ya sea extraídos del texto, de imágenes (facturas), o de audio.
    """
    userintention: Literal["create_provider", "create_client", "create_product", "other", "none", "bye"] = Field(
        ...,
        description=(
            "Intención detectada del usuario. Valores posibles:\n"
            "- 'create_provider': crear un proveedor (datos pueden venir del texto o de una factura)\n"
            "- 'create_client': crear un cliente\n"
            "- 'create_product': crear uno o más productos (datos pueden venir del texto o de items de factura)\n"
            "- 'other': conversación casual u otro propósito\n"
            "- 'none': sin intención clara\n"
            "- 'bye': despedida"
        )
    )
    payload_provider: Optional[PayloadCreateProvider] = Field(
        None, 
        description="Datos del proveedor a crear. Se llena automáticamente si hay una factura en la imagen (usando datos del emisor) o si el usuario proporciona los datos por texto/audio"
    )
    payload_client: Optional[PayloadCreateClient] = Field(
        None,
        description="Datos del cliente a crear. Se llena si el usuario proporciona los datos por texto/audio"
    )
    payload_product: Optional[PayloadCreateProduct] = Field(
        None,
        description="Datos del producto a crear. Se llena automáticamente si hay una factura en la imagen (usando el primer item) o si el usuario proporciona los datos por texto/audio"
    )
    audio_transcription: Optional[str] = Field(
        None,
        description="Transcripción del audio si el usuario envió un mensaje de voz"
    )

class Emisor(BaseModel):
    """Representa al emisor de la factura."""

    razonSocial: str = Field(
        ...,
        description="Nombre o razón social de la empresa o persona que emite la factura."
    )
    nit: str = Field(
        ...,
        description="Número de identificación tributaria (NIT) del emisor, sin dígito de verificación."
    )

    model_config = {
        "title": "Emisor",
        "description": "Entidad o persona responsable de emitir la factura."
    }


class Item(BaseModel):
    """Detalle de un producto o servicio dentro de la factura."""

    descripcion: str = Field(
        ...,
        description="Descripción detallada del producto o servicio facturado."
    )
    cantidad: float = Field(
        ...,
        description="Cantidad de unidades del producto o servicio."
    )
    precioUnitario: Optional[float] = Field(
        None,
        description="Precio unitario del producto o servicio, antes de impuestos o descuentos."
    )
    subtotal: Optional[float] = Field(
        None,
        description="Subtotal del ítem (cantidad × precioUnitario)."
    )

    model_config = {
        "title": "Item de factura",
        "description": "Elemento que describe un producto o servicio incluido en la factura."
    }


class FacturaColombiana(BaseModel):
    """Modelo estructurado de una factura colombiana estándar."""

    numeroFactura: str = Field(
        ...,
        description="Número único de la factura, por ejemplo 'FV-102223'."
    )
    fechaEmision: str = Field(
        ...,
        description="Fecha de emisión de la factura en formato ISO 8601 (YYYY-MM-DD)."
    )
    moneda: str = Field(
        "COP",
        description="Código de moneda en formato ISO 4217. Por defecto 'COP' para pesos colombianos."
    )
    total: float = Field(
        ...,
        description="Valor total de la factura en la moneda especificada."
    )
    emisor: Emisor = Field(
        ...,
        description="Información del emisor de la factura, incluyendo razón social y NIT."
    )
    items: List[Item] = Field(
        default_factory=list,
        description="Lista de ítems o conceptos facturados, cada uno con su descripción, cantidad y valores."
    )

    model_config = {
        "title": "FacturaColombiana",
        "description": "Representa una factura de venta emitida en Colombia conforme a la normativa local, incluyendo información del emisor, ítems facturados y total."
    }
