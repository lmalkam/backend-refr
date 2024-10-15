from typing import Union

from fastapi import FastAPI
from score_route import score_router

app = FastAPI()

app.include_router(score_router)


@app.get("/")
def read_root():
    return {"Hello": "Linky"}