from sqlalchemy import (
    Column,
    String,
    Integer,
    Numeric,
    CheckConstraint,
    UniqueConstraint,
    ForeignKey,
    TIMESTAMP,
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from ..common.base import Base


class Producto(Base):
    __tablename__ = "productos"
    __table_args__ = (
        UniqueConstraint("sku", name="productos_sku_key"),
        CheckConstraint("cantidad >= 0", name="productos_cantidad_check"),
        CheckConstraint("precio_venta >= 0", name="productos_precio_venta_check"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    sku = Column(String(50), nullable=False)
    nombre = Column(String(200), nullable=False)
    precio_venta = Column(Numeric(15, 2), nullable=False)
    cantidad = Column(Integer, nullable=False, default=0)
    proveedor_id = Column(Integer, ForeignKey('terceros.id'), nullable=True)
    fecha_creacion = Column(TIMESTAMP, nullable=True, server_default=func.current_timestamp())
    
    # Relationship to Tercero (provider)
    proveedor = relationship("Tercero", foreign_keys=[proveedor_id])

    def __repr__(self):
        return (
            f"<Producto id={self.id} sku='{self.sku}' nombre='{self.nombre}' "
            f"precio_venta={self.precio_venta} cantidad={self.cantidad}>"
        )