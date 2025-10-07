
from typing import Optional, List, Literal, Any, Dict
from pydantic import BaseModel, Field

class PayloadCreateProvider(BaseModel):
    nombre: Optional[str] = None
    nit: Optional[str] = None
    direccion: Optional[str] = None
    telefono: Optional[str] = None

class PayloadCreateProduct(BaseModel):
    nombre: str = Field(..., description="Nombre del producto", max_length=200)
    precio_venta: float = Field(..., description="Precio de venta del producto", ge=0)
    cantidad: int = Field(0, description="Cantidad disponible del producto", ge=0)
    sku: Optional[str] = Field(None, description="Código SKU del producto (se genera automáticamente si no se proporciona)", max_length=50)
    proveedor: Optional[str] = Field(None, description="Nombre o NIT del proveedor del producto")

class PayloadCreateClient(BaseModel):
    nombre: str = Field(..., description="Nombre del cliente")
    nit: str = Field(..., description="NIT del cliente")
    direccion: str = Field(..., description="Dirección del cliente")


class UserIntention(BaseModel):
    """
    Modelo de salida estructurada: intención + payload correspondiente.
    """
    userintention: Literal["create_provider", "create_client", "create_product", "other", "none", "bye"] = Field(
        ...,
        description=(
            "'create_provider': cuando el usuario quiere crear un proveedor. "
            "'create_client': cuando el usuario quiere crear un cliente. "
            "'create_product': cuando el usuario quiere crear un producto. "
            "'other': conversación casual u otro propósito."
        )
    )
    payload_provider: Optional[PayloadCreateProvider] = None
    payload_client: Optional[PayloadCreateClient] = None
    payload_product: Optional[PayloadCreateProduct] = None
    audio_transcription: Optional[str] = None

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
