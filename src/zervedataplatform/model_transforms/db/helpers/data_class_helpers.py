from dataclasses import field, fields, is_dataclass
from datetime import datetime
from typing import Type, List, Tuple


def primary_key():
    """Helper to mark a field as the primary key."""
    return field(default=None, metadata={"is_pkey": True})


def created_time_stamp():
    """Helper to mark a field as the created time stamp."""
    return field(default=datetime.now().strftime('%Y-%m-%d %H:%M:%S'), metadata={"auto_time_stamp": True})


# TODO fix, this will cause issues since we are ONLY using the data class name to create the table, when we are allowing the user to create
#   custom table names in the db_config!!!
def foreign_key(references_class, references_column: str):
    """Helper to mark a field as a foreign key referencing another dataclass field."""
    return field(default=None, metadata={
        "is_fkey": True,
        "references_class": references_class.__name__.lower(),  # Class name as table name
        "references_column": references_column  # Column name passed as string
    })


def dataclass_to_fields(data_class) -> List[Tuple[str, str]]:
    """Converts a dataclass to a list of field names and types."""
    result = []
    for field in fields(data_class):
        # Check if field.type has a __name__ attribute (if it's a class/type)
        if hasattr(field.type, '__name__'):
            field_type = field.type.__name__
        else:
            # Use str() for cases where __name__ is not available (like generics)
            field_type = str(field.type)
        result.append((field.name, field_type))
    return result


# helper utility
def validate_dataclass_fields_not_none(instance) -> bool:
    """Ensure all fields in the dataclass instance are not None."""
    if not is_dataclass(instance):
        raise ValueError("Provided instance is not a dataclass.")

    for field in fields(instance):
        field_value = getattr(instance, field.name)
        if field_value is None:
            raise ValueError(f"Field '{field.name}' cannot be None.")

    return True  # If all fields are valid
