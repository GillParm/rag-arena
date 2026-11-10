from fastapi import FastAPI

app = FastAPI()


@app.get("/")
async def root():
    # Return a greeting
    return {"message": "Hello World"}
