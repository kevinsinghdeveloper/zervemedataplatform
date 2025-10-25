from dataclasses import dataclass, fields

from zervedataplatform.model_transforms.db.helpers.data_class_helpers import created_time_stamp


@dataclass
class ModelBase:
    createdDate: str = created_time_stamp()  # action_date TIMESTAMPTZ DEFAULT current_timestamp

    @classmethod
    def get_field_name(cls, field_name: str) -> str:
        return field_name

    def get_field_value_by_name(self, field_name: str):
        # Get the list of fields from the dataclass
        try:
            name_field = next(f for f in fields(self) if f.name == field_name)
            return getattr(self, name_field.name)
        except StopIteration:
            return None

    def set_field_value_by_name(self, field_name: str, value):
        # Get the list of fields from the dataclass
        try:
            name_field = next(f for f in fields(self) if f.name == field_name)
            setattr(self, name_field.name, value)
        except StopIteration:
            raise ValueError(f"Field '{field_name}' not found in {self.__class__.__name__}.")