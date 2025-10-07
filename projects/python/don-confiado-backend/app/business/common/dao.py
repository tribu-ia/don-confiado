from typing import TypeVar, Generic, Type, List, Optional
from sqlalchemy.orm import Session

T = TypeVar("T")  # Representa una entidad genérica (modelo SQLAlchemy)

class GenericDAO(Generic[T]):
    def __init__(self, session: Session, model: Type[T]):
        self.session = session
        self.model = model

    def create(self, entity: T) -> T:
        self.session.add(entity)
        self.session.commit()
        self.session.refresh(entity)
        return entity

    def findById(self, id_value) -> Optional[T]:
        return self.session.get(self.model, id_value)

    def findBy(self, **filters) -> Optional[T]:
        return self.session.query(self.model).filter_by(**filters).first()

    def findAll(self) -> List[T]:
        return self.session.query(self.model).all()

    def update(self, id_value, **kwargs) -> Optional[T]:
        entity = self.findById(id_value)
        if entity:
            for key, value in kwargs.items():
                setattr(entity, key, value)
            self.session.commit()
            self.session.refresh(entity)
        return entity

    def delete(self, id_value) -> Optional[T]:
        entity = self.findById(id_value)
        if entity:
            self.session.delete(entity)
            self.session.commit()
        return entity
