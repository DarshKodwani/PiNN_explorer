import os
import yaml
import logging
import streamlit as st
import sys
import streamlit as st

# Import Azure SDK modules
from azure.identity import DefaultAzureCredential
from azure.mgmt.resource import ResourceManagementClient
from azure.mgmt.network import NetworkManagementClient
from azure.mgmt.compute import ComputeManagementClient
from azure.mgmt.storage import StorageManagementClient
from azure.storage.blob import BlobServiceClient
from src.simulators.run_moving_lid_sims_on_azure_vm import create_resource_group, create_storage_account, get_storage_account_key, create_container, create_virtual_network, create_subnet, create_public_ip, create_nic, run_simulation_on_vm, create_vm, upload_script_to_vm, download_results_from_vm, upload_results_to_azure, upload_to_azure

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Streamlit app
st.title("Azure Simulation")

# Upload YAML file
uploaded_file = st.file_uploader("Upload Simulation Input YAML", type=["yaml", "yml"])
if uploaded_file is not None:
    yaml_path = uploaded_file.name
    with open(yaml_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"File {uploaded_file.name} uploaded successfully.")
    
    # Load configuration from YAML file
    with open(yaml_path, 'r') as file:
        config = yaml.safe_load(file)

    subscription_id = config['azure']['subscription_id']
    resource_group_name = config['azure']['resource_group_name']
    location = config['azure']['location']
    storage_account_name = config['azure']['storage_account_name']
    container_name = config['azure']['container_name']
    vm_name = config['azure']['vm_name']
    admin_username = config['azure']['admin_username']
    admin_password = config['azure']['admin_password']

    if st.button("Run Simulation"):
        # Authenticate with Azure
        credential = DefaultAzureCredential()
        resource_client = ResourceManagementClient(credential, subscription_id)
        network_client = NetworkManagementClient(credential, subscription_id)
        compute_client = ComputeManagementClient(credential, subscription_id)
        storage_client = StorageManagementClient(credential, subscription_id)

        # Create resource group and storage account
        create_resource_group(resource_client, resource_group_name, location)
        create_storage_account(storage_client, resource_group_name, storage_account_name, location)

        # Get storage account key and set connection string
        storage_account_key = get_storage_account_key(storage_client, resource_group_name, storage_account_name)
        AZURE_STORAGE_CONNECTION_STRING = f"DefaultEndpointsProtocol=https;AccountName={storage_account_name};AccountKey={storage_account_key};EndpointSuffix=core.windows.net"

        # Create container
        blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
        create_container(blob_service_client, container_name)

        # Create virtual network, subnet, public IP, and network interface
        create_virtual_network(network_client, resource_group_name, location)
        create_subnet(network_client, resource_group_name)
        public_ip = create_public_ip(network_client, resource_group_name, location)
        nic = create_nic(network_client, public_ip, resource_group_name, location, subscription_id)

        # Create virtual machine
        create_vm(compute_client, nic.id, resource_group_name, vm_name, location, admin_username, admin_password)

        # Upload script to VM and run simulation
        upload_script_to_vm(public_ip.ip_address, admin_username)
        run_simulation_on_vm(public_ip.ip_address, admin_username, storage_account_name, container_name)    