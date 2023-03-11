from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
from ml import obtain_image
app = FastAPI()

@app.get("/") #for this path use this function
def read_root():
    return {"Hello": "World"}

@app.get("/items/{item_id}")
def read_item(item_id: int):
    return {"item_id": item_id, "message": "Hello World"}

class Item(BaseModel):
    name: str
    price: float
    tags: list[str] = []

@app.post("/items/")
def create_item(item: Item):
    return item

@app.get("/generate")
def generate_image(
    prompt: str,
    *,
    seed: int = None,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5
):
    image = obtain_image(
        prompt,
        num_inference_steps=num_inference_steps,
        seed=seed,
        guidance_scale=guidance_scale,
    )
    image.save("image.png")
    return FileResponse("image.png")