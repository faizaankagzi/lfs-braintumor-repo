from fastapi import FastAPI, UploadFile
from util import pipeline

# Initializing App
app = FastAPI()

# Allowing Cross Origins
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Schema For Request
from pydantic import BaseModel
class Schema(BaseModel):
	# img_base64:str
    image: UploadFile

print(" ........... App Started ........... ")

# Endpoints

@app.get("/")
def index():
	return "This is the API for Brain Tumor"
      
@app.post("/predict")
def endpoint_face_analytics(image: UploadFile):
	response =  pipeline(image)
	return response

# @app.post("/predict")
# def endpoint_face_analytics(req: Schema):
# 	response =  pipeline(req.img_base64)
# 	return response
