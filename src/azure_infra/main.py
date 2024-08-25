from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import yaml
import os
from fastapi.responses import FileResponse

app = FastAPI()

# Add a root endpoint to handle GET requests to "/"
@app.get("/")
async def read_root():
    return {"message": "Welcome to the FastAPI application! The server is running."}

# Define a Pydantic model for the request body
class NumberInput(BaseModel):
    number: int

# Function to analyze the YAML file
def analyze_yaml(yaml_content):
    # Example analysis: count the number of keys in the YAML file
    return {"key_count": len(yaml_content)}

# New POST endpoint to handle YAML file upload and analysis
@app.post("/analyze-yaml/")
async def analyze_yaml_file(file: UploadFile = File(...)):
    content = await file.read()
    yaml_content = yaml.safe_load(content)
    analysis_result = analyze_yaml(yaml_content)
    return analysis_result

# Function to generate a text file from YAML content
def generate_text_file(yaml_content, file_path):
    with open(file_path, 'w') as f:
        for key, value in yaml_content.items():
            f.write(f"{key}: {value}\n")

# New POST endpoint to handle YAML file upload and generate a text file
@app.post("/generate-text/")
async def generate_text(file: UploadFile = File(...)):
    try:
        content = await file.read()
        yaml_content = yaml.safe_load(content)
        
        # Define the path for the generated text file
        file_path = "output.txt"
        
        # Generate the text file
        generate_text_file(yaml_content, file_path)
        
        return {"file_path": file_path}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Endpoint to download the generated text file
@app.get("/download-text/")
async def download_text(file_path: str):
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type='text/plain', filename=os.path.basename(file_path))
    else:
        raise HTTPException(status_code=404, detail="File not found")