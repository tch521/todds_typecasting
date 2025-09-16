# todds_typecasting
Python utilities to cast data according to a specific set of rules. 

The main features of this are the CustomCastingDataclass and the auto_cast function decorator.

What separates this from pydantic is ease-of-use and some custom treatment to make use of `json.loads()`.

It's almostly entirely AI generated so use at your own risk. 

This will likely be refactored to use pydantic in the future. 
Or maybe I'll just contribute to pydantic directly.
I feel like the auto_cast decorator in particular is a no-brainer, so much easier than building a whole pydantic model for simple functions.

## Example Usage
```python
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Union

from todds_typecasting.todds_typecasting import CustomCastingDataclass, auto_cast

# --- Dataclass example (demonstrates: datetime, list[bool], tuple[int,...], dict with Union value types,
#     Union ordering behavior, Optional[float], JSON strings, tuple -> list normalization) ---

@dataclass
class AppConfig(CustomCastingDataclass):
    port: int
    debug: bool
    flags: list[bool]
    thresholds: tuple[int, ...]
    started_at: datetime
    metadata: dict[str, Union[int, float, str]]
    mode: Union[str, int]              # Order matters: Union[str, int] keeps "7" as a string
    threshold: Optional[float] = None  # Optional field

cfg = AppConfig(
    port="8080",                        # str -> int
    debug="YES",                        # flexible bool tokens
    flags='["true","0","yes"]',         # JSON string -> list[bool] => [True, False, True]
    thresholds=("1", "2", "3"),         # tuple of str -> tuple[int,...]
    started_at="2024-05-05T10:00:00",   # ISO8601 -> datetime
    metadata='{"retries":"3","ratio":"0.75","note":"ok"}',  # JSON -> dict[str, Union[int,float,str]]
    mode="7",                           # Union[str, int] => stays "7" (first arm wins)
    threshold="0.92",                   # Optional[float] with str input
)

# Resulting object (Python repr):
# AppConfig(
#   port=8080,
#   debug=True,
#   flags=[True, False, True],
#   thresholds=(1, 2, 3),
#   started_at=datetime(2024, 5, 5, 10, 0, 0),
#   metadata={'retries': 3, 'ratio': 0.75, 'note': 'ok'},
#   mode='7',
#   threshold=0.92
# )

# --- Decorator example (auto_cast) ---
# Shows: list[int] from tuple of str, Optional[datetime], Optional[list[int]],
# dict with Union value types from JSON string, bool tokens, JSON list for tags.

@auto_cast
def launch_job(
    attempts: int,
    debug: bool = False,
    schedule_at: Optional[datetime] = None,
    limits: Optional[list[int]] = None,
    tags: list[str] = None,
    attributes: Optional[dict[str, Union[int, float, bool]]] = None,
):
    return {
        "attempts": attempts,
        "debug": debug,
        "schedule_at": schedule_at,
        "limits": limits,
        "tags": tags,
        "attributes": attributes,
        "types": {
            "attempts": type(attempts).__name__,
            "debug": type(debug).__name__,
            "schedule_at": type(schedule_at).__name__ if schedule_at else None,
            "limits": type(limits).__name__ if limits else None,
            "tags": type(tags).__name__,
            "attributes": type(attributes).__name__ if attributes else None,
        },
    }

result = launch_job(
    attempts="5",
    debug="true",
    schedule_at="2025-01-01T00:00:00",
    limits=("1", "2", "3"),                         # tuple -> list[int]
    tags='["alpha","beta"]',                        # JSON -> list[str]
    attributes='{"retries":"3","success":"true","ratio":"0.9"}',  # JSON -> dict[str, Union[int,bool,float]]
)

# Returned (illustrative):
# {
#   'attempts': 5,
#   'debug': True,
#   'schedule_at': datetime(2025, 1, 1, 0, 0, 0),
#   'limits': [1, 2, 3],
#   'tags': ['alpha', 'beta'],
#   'attributes': {'retries': 3, 'success': True, 'ratio': 0.9},
#   'types': {
#       'attempts': 'int',
#       'debug': 'bool',
#       'schedule_at': 'datetime',
#       'limits': 'list',
#       'tags': 'list',
#       'attributes': 'dict'
#   }
# }
```


