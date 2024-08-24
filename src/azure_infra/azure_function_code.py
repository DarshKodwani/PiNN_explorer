import os
import zipfile
import shutil
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential
from azure.mgmt.resource import ResourceManagementClient
from azure.mgmt.web import WebSiteManagementClient
from azure.storage.blob import BlobServiceClient
from azure.core.exceptions import ResourceNotFoundError
import requests
import time

# Load environment variables from .env file (if you have any environment-specific settings)
load_dotenv('.env')

# Azure configuration settings
subscription_id = os.getenv('subscription_id')
resource_group_name = "pinns"
location = "uksouth"
storage_account_name = "pinns"
app_service_plan_name = "pinns"
web_app_name = "pinns"

# File names
simulation_script = "moving_lid_sims.py"
input_yaml = "sims_inputs.yaml"
upload_directory = "uploads"
requirements_file = "requirements.txt"

# FastAPI script content
fastapi_script = """
from fastapi import FastAPI, UploadFile, File, HTTPException
import yaml
import shutil
import os

from moving_lid_sims import run_simulation  # Replace with the actual function in your simulation script

app = FastAPI()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/run-simulation/")
async def run_simulation_api(file: UploadFile = File(...)):
    try:
        # Save uploaded YAML file
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Read and parse YAML file
        with open(file_path, 'r') as stream:
            input_data = yaml.safe_load(stream)

        # Run simulation with parsed input data
        result = run_simulation(input_data)

        return {"message": "Simulation completed successfully.", "result": result}
    except yaml.YAMLError as e:
        raise HTTPException(status_code=400, detail=f"Invalid YAML file: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during simulation: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
"""

# Requirements file content
requirements_content = """
fastapi
uvicorn
pyyaml
"""

# Initialize Azure clients
credential = DefaultAzureCredential()
resource_client = ResourceManagementClient(credential, subscription_id)
web_client = WebSiteManagementClient(credential, subscription_id)

# Create or get resource group
try:
    resource_client.resource_groups.get(resource_group_name)
    print(f"Resource group {resource_group_name} already exists.")
except ResourceNotFoundError:
    resource_client.resource_groups.create_or_update(resource_group_name, {"location": location})
    print(f"Resource group {resource_group_name} created.")

# Create or get App Service Plan
try:
    app_service_plan = web_client.app_service_plans.get(resource_group_name, app_service_plan_name)
    print(f"App service plan {app_service_plan_name} already exists.")
except ResourceNotFoundError:
    app_service_plan = web_client.app_service_plans.begin_create_or_update(
        resource_group_name,
        app_service_plan_name,
        {
            "location": location,
            "sku": {"name": "B1", "tier": "Basic"},  # Adjust the SKU based on your needs
        },
    ).result()
    print(f"App service plan {app_service_plan_name} created.")

# Create or get Web App
try:
    web_client.web_apps.get(resource_group_name, web_app_name)
    print(f"Web app {web_app_name} already exists.")
except ResourceNotFoundError:
    web_app = web_client.web_apps.begin_create_or_update(
        resource_group_name,
        web_app_name,
        {
            "location": location,
            "server_farm_id": app_service_plan.id,
            "site_config": {
                "app_settings": [
                    {"name": "WEBSITE_RUN_FROM_PACKAGE", "value": f"https://{storage_account_name}.blob.core.windows.net/$web/webapp.zip"},
                    {"name": "FUNCTIONS_WORKER_RUNTIME", "value": "python"},  # Optional if using Functions runtime
                ]
            },
        },
    ).result()
    print(f"Web app {web_app_name} created successfully.")

# Create application directory structure
os.makedirs(upload_directory, exist_ok=True)

# Write FastAPI script to app directory
with open("app.py", "w") as f:
    f.write(fastapi_script)

# Write requirements.txt to app directory
with open(requirements_file, "w") as f:
    f.write(requirements_content)

# Copy simulation script to app directory
shutil.copy(simulation_script, simulation_script)

# Create a ZIP package for deployment
zipf = zipfile.ZipFile("webapp.zip", "w", zipfile.ZIP_DEFLATED)
zipf.write("app.py")
zipf.write(requirements_file)
zipf.write(simulation_script)
zipf.close()

# Deploy the ZIP file to Azure App Service
blob_service_client = BlobServiceClient.from_connection_string(connection_string)
blob_client = blob_service_client.get_blob_client(container="$web", blob="webapp.zip")
with open("webapp.zip", "rb") as data:
    blob_client.upload_blob(data, overwrite=True)

# Adding a delay to allow Azure to deploy the app
print("Waiting for the web app to be fully deployed...")
time.sleep(60)  # Wait for 60 seconds (adjust as needed)

# Check if the web app is working
web_app_url = f"https://{web_app_name}.azurewebsites.net/run-simulation/"
try:
    response = requests.post(web_app_url, files={'file': open(input_yaml, 'rb')})
    if response.status_code == 200:
        print("Web app is up and running successfully!")
        print("Response from the app:", response.json())
    else:
        print(f"Web app responded with status code: {response.status_code}")
        print("Response text:", response.text)
except requests.exceptions.RequestException as e:
    print(f"An error occurred while trying to reach the web app: {e}")