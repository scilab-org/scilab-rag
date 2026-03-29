"""
Shared base model that enforces snake_case internally and camelCase on the wire.

All request and response schemas inherit from CamelCaseModel so that:
- Python code always uses snake_case attribute access.
- JSON sent to / received from external clients uses camelCase keys.
"""

from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel


class CamelCaseModel(BaseModel):
    model_config = ConfigDict(
        alias_generator=to_camel,   # snake_case → camelCase for all fields
        populate_by_name=True,      # also allow construction by the snake_case name
        from_attributes=True,       # allow ORM-mode validation
    )
