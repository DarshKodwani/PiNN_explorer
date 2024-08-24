import os
import time
import zipfile
import requests
from azure.identity import DefaultAzureCredential
from azure.mgmt.storage import StorageManagementClient
from azure.mgmt.storage.models import StorageAccountCreateParameters, Sku, Kind
from azure.storage.blob import BlobServiceClient

# Replace with your subscription ID, resource group name, and storage account name
subscription_id = os.getenv('subscription_id')
resource_group_name = 'your_resource_group_name'
storage_account_name = 'your_storage_account_name'
location = 'your_location'
web_app_name = 'your_web_app_name'
input_yaml = 'path_to_your_input_yaml_file'

# Create a DefaultAzureCredential object
credential = DefaultAzureCredential()

# Create a StorageManagementClient object
storage_client = StorageManagementClient(credential, subscription_id)

# Create the storage account
storage_async_operation = storage_client.storage_accounts.begin_create(
    resource_group_name,
    storage_account_name,
    StorageAccountCreateParameters(
        sku=Sku(name='Standard_LRS'),
        kind=Kind.STORAGE_V2,
        location=location
    )
)
storage_account = storage_async_operation.result()

# Retrieve the storage account keys
keys = storage_client.storage_accounts.list_keys(resource_group_name, storage_account_name)
storage_keys = {v.key_name: v.value for v in keys.keys}

# Construct the connection string
connection_string = (
    f"DefaultEndpointsProtocol=https;"
    f"AccountName={storage_account_name};"
    f"AccountKey={storage_keys['key1']};"
    f"EndpointSuffix=core.windows.net"
)

# Write requirements.txt to app directory
requirements_file = 'requirements.txt'
requirements_content = 'your_requirements_content_here'
with open(requirements_file, "w") as f:
    f.write(requirements_content)

# Copy simulation script to app directory
simulation_script = 'path_to_your_simulation_script'
# shutil.copy(simulation_script, simulation_script)

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
    print(f"An error occurred: {e}")