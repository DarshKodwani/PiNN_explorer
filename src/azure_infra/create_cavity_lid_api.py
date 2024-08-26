from fastapi import FastAPI, UploadFile, File, HTTPException
import subprocess
import yaml
import os
from dotenv import load_dotenv
from pydantic import BaseModel
from src.simulators.moving_lid_sims import moving_lid_sim_runner
from fastapi.responses import FileResponse

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
        csv_file, plot_file, animation_file = moving_lid_sim_runner(yaml_content)
        
        return {"csv_path": csv_file, "plot_path": plot_file, "animation_path": animation_file}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
#Endpoint to download the generated csv file
@app.get("/download-csv/")
async def download_csv(file_path: str):
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type='text/csv', filename=os.path.basename(file_path))
    else:
        raise HTTPException(status_code=404, detail="File not found")

#Endpoint to download the generated plot file
@app.get("/download-plot/")
async def download_plot(file_path: str):
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type='image/png', filename=os.path.basename(file_path))
    else:
        raise HTTPException(status_code=404, detail="File not found")

#Endpoint to download the generated animation file
@app.get("/download-animation/")
async def download_animation(file_path: str):
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type='video/mp4', filename=os.path.basename(file_path))
    else:
        raise HTTPException(status_code=404, detail="File not found")