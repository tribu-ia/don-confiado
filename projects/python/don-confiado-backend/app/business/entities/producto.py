from sqlalchemy import (
    Column,
    String,
    Integer,
    Text,
    Numeric,
    Boolean,
    Date,
    CheckConstraint,
    UniqueConstraint,
    TIMESTAMP,
)
from sqlalchemy.sql import func
from ..common.base import Base


class Producto(Base):
    __tablename__ = "productos"
    __table_args__ = (
        UniqueConstraint("sku", name="productos_sku_key"),
        UniqueConstraint("ean_code", name="productos_ean_code_key"),
        CheckConstraint("stock >= 0", name="productos_stock_check"),
        CheckConstraint("stock_minimo >= 0", name="productos_stock_minimo_check"),
        CheckConstraint("costo_ultimo >= 0", name="productos_costo_ultimo_check"),
        CheckConstraint("precio_promedio_compra >= 0", name="productos_precio_promedio_compra_check"),
        CheckConstraint("precio_venta >= 0", name="productos_precio_venta_check"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    sku = Column(String(50), nullable=False)
    nombre = Column(String(200), nullable=False)
    descripcion = Column(Text, nullable=True)
    marca = Column(String(100), nullable=True)
    categoria = Column(String(100), nullable=True)
    ean_code = Column(String(50), nullable=True)
    precio_venta = Column(Numeric(15, 2), nullable=False)
    precio_promedio_compra = Column(Numeric(15, 2), nullable=True)
    costo_ultimo = Column(Numeric(15, 2), nullable=True)
    stock = Column(Integer, nullable=False, default=0)
    stock_minimo = Column(Integer, nullable=False, default=0)
    ubicacion_almacen = Column(String(100), nullable=True)
    unidad_medida = Column(String(20), nullable=False, default="UND")
    activo = Column(Boolean, nullable=False, default=True)
    perecedero = Column(Boolean, nullable=False, default=False)
    fecha_vencimiento = Column(Date, nullable=True)
    fecha_creacion = Column(TIMESTAMP, nullable=True, server_default=func.current_timestamp())

    def __repr__(self):
        return (
            f"<Producto id={self.id} sku='{self.sku}' nombre='{self.nombre}' "
            f"precio_venta={self.precio_venta} stock={self.stock}>"
        )