from fastapi import FastAPI
from app.api.controller import router as chat_router
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="PNB_COREAI_BACKEND")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or "*"
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the routes from the controller
app.include_router(chat_router, prefix="/api/v1")

@app.get("/")
async def root():
    return {"message": "AI Backend is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)