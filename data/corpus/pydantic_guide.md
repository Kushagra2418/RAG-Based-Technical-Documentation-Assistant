# Pydantic v2 Reference Guide

Pydantic is the most widely used data-validation library for Python. It uses Python type
hints to validate, parse, and serialize data. Version 2 rewrote the validation core in
Rust (the `pydantic-core` package), making it significantly faster than version 1.

## Defining a Model

A model is a class that inherits from `BaseModel`. Fields are declared as annotated
class attributes:

```python
from pydantic import BaseModel

class User(BaseModel):
    id: int
    name: str
    is_active: bool = True
```

Creating an instance validates and coerces the input:

```python
user = User(id="123", name="Ada")
# user.id == 123  (the string "123" is coerced to an int)
# user.is_active == True  (default value applied)
```

If validation fails, Pydantic raises a `ValidationError` that lists every problem at
once, including the location, message, and type of each error.

## Field Constraints

The `Field` function adds metadata and validation constraints to a field:

```python
from pydantic import BaseModel, Field

class Product(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    price: float = Field(gt=0, description="Price in USD")
    quantity: int = Field(default=0, ge=0)
```

Common numeric constraints are `gt` (greater than), `ge` (greater than or equal),
`lt` (less than), and `le` (less than or equal). String constraints include
`min_length`, `max_length`, and `pattern` (a regular expression).

## Optional and Default Values

A field is required unless it has a default. Use `Optional` or the `| None` union to
allow `None`, and provide a default to make it optional:

```python
from typing import Optional

class Config(BaseModel):
    timeout: int = 30            # optional, defaults to 30
    label: Optional[str] = None  # optional, may be None
```

For mutable defaults such as lists or dicts, use `default_factory` to avoid sharing a
single instance across objects:

```python
from pydantic import Field

class Basket(BaseModel):
    items: list[str] = Field(default_factory=list)
```

## Validators

Field validators run custom logic for one or more fields. In Pydantic v2 they are
declared with the `field_validator` decorator:

```python
from pydantic import BaseModel, field_validator

class Account(BaseModel):
    email: str

    @field_validator("email")
    @classmethod
    def must_contain_at(cls, value: str) -> str:
        if "@" not in value:
            raise ValueError("email must contain @")
        return value
```

Model validators validate the object as a whole, after all fields are set, using the
`model_validator` decorator with `mode="after"`.

## Serialization

Pydantic v2 replaced v1's `.dict()` and `.json()` with `.model_dump()` and
`.model_dump_json()`:

```python
user = User(id=1, name="Ada")
user.model_dump()        # -> {"id": 1, "name": "Ada", "is_active": True}
user.model_dump_json()   # -> '{"id":1,"name":"Ada","is_active":true}'
```

Use the `exclude`, `include`, and `exclude_none` arguments to control which fields are
serialized.

## Parsing Untrusted Input

To build a model from arbitrary data (for example, a decoded JSON payload), use
`model_validate` for dictionaries and `model_validate_json` for raw JSON strings:

```python
User.model_validate({"id": 1, "name": "Ada"})
User.model_validate_json('{"id": 1, "name": "Ada"}')
```

## Settings Management

The companion package `pydantic-settings` reads configuration from environment variables
and `.env` files into a typed settings object:

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    api_key: str
    debug: bool = False

    model_config = {"env_file": ".env"}

settings = Settings()
```

Each field maps to an environment variable of the same name (case-insensitive). This is
the recommended way to manage secrets and configuration in production applications.

## Migration Notes from v1

- `.dict()` becomes `.model_dump()`.
- `.json()` becomes `.model_dump_json()`.
- `@validator` becomes `@field_validator`.
- `class Config` becomes the `model_config` dict (or `SettingsConfigDict`).
- `parse_obj` becomes `model_validate`.
