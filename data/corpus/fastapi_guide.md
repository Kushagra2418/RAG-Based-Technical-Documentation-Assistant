# FastAPI Reference Guide

FastAPI is a modern, high-performance web framework for building APIs with Python,
based on standard Python type hints. It is built on top of Starlette (for the web
parts) and Pydantic (for the data parts).

## Why FastAPI

- **Fast**: Very high performance, on par with NodeJS and Go, thanks to Starlette and
  async support.
- **Fast to code**: Type hints and editor autocompletion reduce development time.
- **Fewer bugs**: Type checking and automatic validation catch many errors early.
- **Automatic docs**: Interactive API documentation (Swagger UI and ReDoc) is generated
  automatically from your code.

## A Minimal Application

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello World"}
```

Run it with an ASGI server such as Uvicorn:

```bash
uvicorn main:app --reload
```

The `--reload` flag restarts the server automatically when code changes. It should only
be used during development, never in production.

## Path Parameters

Path parameters are declared with Python format-string syntax and are validated using
type hints:

```python
@app.get("/items/{item_id}")
def read_item(item_id: int):
    return {"item_id": item_id}
```

Because `item_id` is annotated as `int`, FastAPI converts and validates the incoming
value. A request to `/items/abc` returns a clear `422 Unprocessable Entity` error, while
`/items/3` yields `item_id` as the integer `3`.

## Query Parameters

Function parameters that are not part of the path are interpreted as query parameters:

```python
@app.get("/items/")
def list_items(skip: int = 0, limit: int = 10):
    return {"skip": skip, "limit": limit}
```

A request to `/items/?skip=20&limit=50` sets the two values accordingly. Parameters with
default values are optional; parameters without defaults are required.

## Request Body

To receive a JSON request body, declare a Pydantic model as a parameter:

```python
from pydantic import BaseModel

class Item(BaseModel):
    name: str
    price: float
    is_offer: bool = False

@app.post("/items/")
def create_item(item: Item):
    return {"name": item.name, "price": item.price}
```

FastAPI reads the body as JSON, validates it against the model, and provides a fully
typed `item` object. Invalid bodies produce a `422` response describing exactly which
field failed.

## Response Models

Use the `response_model` argument to declare and filter the response shape:

```python
@app.post("/items/", response_model=Item)
def create_item(item: Item):
    return item
```

FastAPI uses the response model to serialize output, validate it, and document it. Any
fields not present in the response model are stripped, which is useful for hiding
sensitive fields such as passwords.

## Status Codes

Set the default status code for an operation with `status_code`:

```python
from fastapi import status

@app.post("/items/", status_code=status.HTTP_201_CREATED)
def create_item(item: Item):
    return item
```

## Handling Errors

Raise `HTTPException` to return an HTTP error with a specific status code and detail:

```python
from fastapi import HTTPException

@app.get("/items/{item_id}")
def read_item(item_id: int):
    if item_id not in items:
        raise HTTPException(status_code=404, detail="Item not found")
    return items[item_id]
```

## File Uploads

Use `UploadFile` for file uploads. `UploadFile` streams large files to disk instead of
holding them entirely in memory:

```python
from fastapi import UploadFile

@app.post("/upload/")
async def upload(file: UploadFile):
    content = await file.read()
    return {"filename": file.filename, "size": len(content)}
```

To accept both files and form fields in one request, combine `UploadFile` with `Form`.

## Dependency Injection

FastAPI has a first-class dependency-injection system through the `Depends` helper.
Dependencies can provide shared resources, enforce authentication, or supply common
parameters:

```python
from fastapi import Depends

def pagination(skip: int = 0, limit: int = 10):
    return {"skip": skip, "limit": limit}

@app.get("/users/")
def list_users(page: dict = Depends(pagination)):
    return page
```

## Async vs Sync

Path operation functions may be defined with `def` or `async def`. Use `async def` when
you call libraries that support `await`. Use plain `def` for blocking libraries; FastAPI
runs those in an external threadpool so they do not block the event loop.

## Interactive Documentation

Once the server is running, interactive documentation is available at `/docs`
(Swagger UI) and `/redoc` (ReDoc). Both are generated from the OpenAPI schema that
FastAPI builds automatically from your type hints and models.
