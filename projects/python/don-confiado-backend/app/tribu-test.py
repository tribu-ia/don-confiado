from business.common.connection import SessionLocal, init_db
from business.entities.tercero import Tercero
from business.dao.tercero_dao import TerceroDAO



session = SessionLocal()

dao = TerceroDAO(session)

terceros = dao.findAll()

for t in terceros:
    print(t)