from sqlalchemy import Column, String, Integer, Text, CheckConstraint, UniqueConstraint, TIMESTAMP
from sqlalchemy.sql import func
from ..common.base import Base

class Tercero(Base):
    __tablename__ = "terceros"
    __table_args__ = (
        UniqueConstraint('tipo_documento', 'numero_documento', name='uq_documento'),
        CheckConstraint("tipo_documento IN ('CC', 'NIT', 'CE')", name='terceros_tipo_documento_check'),
        CheckConstraint("tipo_tercero IN ('cliente', 'proveedor', 'empleado')", name='terceros_tipo_tercero_check'),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    tipo_documento = Column(String(5), nullable=False)
    numero_documento = Column(String(30), nullable=False)
    razon_social = Column(String(200), nullable=True)
    nombres = Column(String(100), nullable=True)
    apellidos = Column(String(100), nullable=True)
    telefono_fijo = Column(String(20), nullable=True)
    telefono_celular = Column(String(20), nullable=True)
    tipo_tercero = Column(String(20), nullable=False)
    direccion = Column(Text, nullable=True)
    email = Column(String(150), nullable=True)
    email_facturacion = Column(String(150), nullable=True)
    fecha_creacion = Column(TIMESTAMP, nullable=True, server_default=func.current_timestamp())

    def __repr__(self):
        return (
            f"<Tercero id={self.id} tipo_documento='{self.tipo_documento}' "
            f"numero_documento='{self.numero_documento}' tipo_tercero='{self.tipo_tercero}'>"
        )
