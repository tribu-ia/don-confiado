from business.entities.tercero import Tercero
from business.common.dao import GenericDAO

class TerceroDAO(GenericDAO[Tercero]):
    def __init__(self, session):
        super().__init__(session, Tercero)

    def findByNumeroDocumento(self, numero_documento: str):
        return self.findBy(numero_documento=numero_documento)
