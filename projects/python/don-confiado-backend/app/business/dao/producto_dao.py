from business.entities.producto import Producto
from business.common.dao import GenericDAO
from typing import List, Optional

class ProductoDAO(GenericDAO[Producto]):
    def __init__(self, session):
        super().__init__(session, Producto)

    def findBySku(self, sku: str) -> Optional[Producto]:
        """Find a product by SKU."""
        return self.findBy(sku=sku)
    
    def findByProveedor(self, proveedor_id: int) -> List[Producto]:
        """Find all products from a specific provider."""
        return self.session.query(Producto).filter(
            Producto.proveedor_id == proveedor_id
        ).all()
        
    def findByNombre(self, nombre: str) -> List[Producto]:
        """Find all products by name."""
        return self.session.query(Producto).filter(
            Producto.nombre.ilike(f"%{nombre}%")
        ).all()
    


