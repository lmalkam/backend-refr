from typing import Union

from fastapi import FastAPI
from score_route import score_router
from custombio_route import BioRouter

app = FastAPI()

app.include_router(score_router)
app.include_router(BioRouter)

@app.get("/")
def read_root():
    return {"Hello": "Linky"}