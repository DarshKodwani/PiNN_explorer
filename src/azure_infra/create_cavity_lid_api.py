from fastapi import FastAPI, UploadFile, File, HTTPException
import subprocess
import yaml
import os
from dotenv import load_dotenv
from pydantic import BaseModel

app = FastAPI()

# Add a root endpoint to handle GET requests to "/"
@app.get("/")
async def read_root():
    return {"message": "Welcome to the FastAPI application for cavity lid simulations! The server is running."}

# New POST endpoint to handle YAML file upload and run cavity_lid_simulations
@app.post("/run-simulation/")
async def run_simulation(file: UploadFile = File(...)):
    try:
        content = await file.read()
        yaml_content = yaml.safe_load(content)
        
        # Run the cavity_lid_simulations function with the parsed YAML content
        simulation_result = cavity_lid_simulations(yaml_content)
        
        return {"simulation_result": simulation_result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))