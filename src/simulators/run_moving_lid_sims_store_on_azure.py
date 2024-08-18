import os
from dotenv import load_dotenv
import subprocess
import logging
import numpy as np
from azure.identity import DefaultAzureCredential
from azure.mgmt.resource import ResourceManagementClient
from azure.mgmt.storage import StorageManagementClient
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient

# Prompt for Azure configuration
load_dotenv('.env')
subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID")
resource_group_name = input("Enter your Azure resource group name: ")
location = input("Enter your Azure location (e.g., eastus): ")
storage_account_name = input("Enter your Azure storage account name: ")
container_name = input("Enter your Azure container name: ")

# Azure Storage account details
AZURE_STORAGE_CONNECTION_STRING = None

def create_resource_group(resource_client):
    """
    Creates a resource group.
    """
    resource_client.resource_groups.create_or_update(
        resource_group_name,
        {"location": location}
    )
    print(f"Resource group '{resource_group_name}' created.")

def create_storage_account(storage_client):
    """
    Creates a storage account.
    """
    storage_async_operation = storage_client.storage_accounts.begin_create(
        resource_group_name,
        storage_account_name,
        {
            "location": location,
            "sku": {"name": "Standard_LRS"},
            "kind": "StorageV2",
            "properties": {}
        }
    )
    storage_async_operation.result()
    print(f"Storage account '{storage_account_name}' created.")

def get_storage_account_key(storage_client):
    """
    Retrieves the storage account key.
    """
    keys = storage_client.storage_accounts.list_keys(resource_group_name, storage_account_name)
    return keys.keys[0].value

def create_container(blob_service_client):
    """
    Creates a container in the storage account.
    """
    container_client = blob_service_client.create_container(container_name)
    print(f"Container '{container_name}' created.")

def upload_to_azure(local_file_path, blob_name):
    """
    Uploads a file to Azure Blob Storage.

    Parameters:
    local_file_path (str): Path to the local file.
    blob_name (str): Name of the blob in Azure.
    """
    blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)

    print('Uploading:', local_file_path, 'to', blob_name)  # Debugging statement
    with open(local_file_path, "rb") as data:
        blob_client.upload_blob(data, overwrite=True)

def run_simulation():
    """
    Runs the moving_lid_sims.py script.
    """
    subprocess.run(["python", "src/simulators/moving_lid_sims.py"])

def upload_results(output_dir):
    """
    Uploads all files in the output directory to Azure Blob Storage.

    Parameters:
    output_dir (str): Path to the output directory.
    """
    #print(f"Uploading results from directory: {output_dir}")  # Debugging statement
    for root, dirs, files in os.walk(output_dir):
        print('root:', root)
        for file in files:
            print('file:', file)
            local_file_path = os.path.join(root, file)
            blob_name = os.path.relpath(local_file_path, output_dir)
            print('Uploading:', local_file_path, 'to', blob_name)
            upload_to_azure(local_file_path, blob_name)

if __name__ == "__main__":
    # Authenticate with Azure
    credential = DefaultAzureCredential()
    resource_client = ResourceManagementClient(credential, subscription_id)
    storage_client = StorageManagementClient(credential, subscription_id)

    # Create resource group and storage account
    create_resource_group(resource_client)
    create_storage_account(storage_client)

    # Get storage account key and set connection string
    storage_account_key = get_storage_account_key(storage_client)
    AZURE_STORAGE_CONNECTION_STRING = f"DefaultEndpointsProtocol=https;AccountName={storage_account_name};AccountKey={storage_account_key};EndpointSuffix=core.windows.net"

    # Create container
    blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
    create_container(blob_service_client)

    # Run the simulation
    run_simulation()

    # Define the output directory
    output_dir = os.path.join("simulation_outputs", "moving_lid_simulation")

    # Upload results to Azure
    print('Uploading results to Azure...')
    upload_results(output_dir)