import os
import subprocess
import logging
from azure.identity import DefaultAzureCredential
from azure.mgmt.resource import ResourceManagementClient
from azure.mgmt.compute import ComputeManagementClient
from azure.mgmt.compute.models import DiskCreateOption
from azure.mgmt.network import NetworkManagementClient
from azure.mgmt.network.models import NetworkSecurityGroup, SecurityRule, NetworkInterface, NetworkInterfaceIPConfiguration, PublicIPAddress, VirtualNetwork, Subnet, NetworkSecurityGroup
from azure.storage.blob import BlobServiceClient
from azure.common.credentials import ServicePrincipalCredentials
from azure.mgmt.compute.models import VirtualMachine, HardwareProfile, StorageProfile, OSProfile, NetworkProfile, ImageReference, LinuxConfiguration, SshConfiguration, SshPublicKey

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prompt for Azure configuration
subscription_id = input("Enter your Azure subscription ID: ")
resource_group_name = input("Enter your Azure resource group name: ")
location = input("Enter your Azure location (e.g., eastus): ")
storage_account_name = input("Enter your Azure storage account name: ")
container_name = input("Enter your Azure container name: ")
vm_name = input("Enter your Azure VM name: ")
admin_username = input("Enter the admin username for the VM: ")
admin_password = input("Enter the admin password for the VM: ")

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
    logger.info(f"Resource group '{resource_group_name}' created.")

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
    logger.info(f"Storage account '{storage_account_name}' created.")

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
    logger.info(f"Container '{container_name}' created.")

def create_virtual_network(network_client):
    """
    Creates a virtual network.
    """
    vnet_params = {
        "location": location,
        "address_space": {
            "address_prefixes": ["10.0.0.0/16"]
        }
    }
    network_client.virtual_networks.begin_create_or_update(resource_group_name, "vnet", vnet_params).result()
    logger.info(f"Virtual network 'vnet' created.")

def create_subnet(network_client):
    """
    Creates a subnet.
    """
    subnet_params = {
        "address_prefix": "10.0.0.0/24"
    }
    network_client.subnets.begin_create_or_update(resource_group_name, "vnet", "subnet", subnet_params).result()
    logger.info(f"Subnet 'subnet' created.")

def create_public_ip(network_client):
    """
    Creates a public IP address.
    """
    public_ip_params = {
        "location": location,
        "public_ip_allocation_method": "Dynamic"
    }
    return network_client.public_ip_addresses.begin_create_or_update(resource_group_name, "publicIP", public_ip_params).result()

def create_nic(network_client, public_ip):
    """
    Creates a network interface.
    """
    nic_params = {
        "location": location,
        "ip_configurations": [{
            "name": "ipconfig1",
            "public_ip_address": public_ip,
            "subnet": {
                "id": f"/subscriptions/{subscription_id}/resourceGroups/{resource_group_name}/providers/Microsoft.Network/virtualNetworks/vnet/subnets/subnet"
            }
        }]
    }
    return network_client.network_interfaces.begin_create_or_update(resource_group_name, "nic", nic_params).result()

def create_vm(compute_client, nic_id):
    """
    Creates a virtual machine.
    """
    vm_params = {
        "location": location,
        "hardware_profile": {
            "vm_size": "Standard_DS1_v2"
        },
        "storage_profile": {
            "image_reference": {
                "publisher": "Canonical",
                "offer": "UbuntuServer",
                "sku": "18.04-LTS",
                "version": "latest"
            },
            "os_disk": {
                "name": f"{vm_name}_os_disk",
                "caching": "ReadWrite",
                "create_option": "FromImage",
                "managed_disk": {
                    "storage_account_type": "Standard_LRS"
                }
            }
        },
        "os_profile": {
            "computer_name": vm_name,
            "admin_username": admin_username,
            "admin_password": admin_password,
            "linux_configuration": {
                "disable_password_authentication": False
            }
        },
        "network_profile": {
            "network_interfaces": [{
                "id": nic_id,
                "properties": {
                    "primary": True
                }
            }]
        }
    }
    compute_client.virtual_machines.begin_create_or_update(resource_group_name, vm_name, vm_params).result()
    logger.info(f"Virtual machine '{vm_name}' created.")

def upload_script_to_vm(public_ip_address):
    """
    Uploads the simulation script to the VM.
    """
    os.system(f"scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null src/simulators/moving_lid_sims.py {admin_username}@{public_ip_address}:~/")
    logger.info(f"Simulation script uploaded to VM.")

def run_simulation_on_vm(public_ip_address):
    """
    Runs the simulation script on the VM.
    """
    os.system(f"ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null {admin_username}@{public_ip_address} 'python3 ~/moving_lid_sims.py'")
    logger.info(f"Simulation script executed on VM.")

def download_results_from_vm(public_ip_address):
    """
    Downloads the results from the VM.
    """
    os.system(f"scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null {admin_username}@{public_ip_address}:~/simulation_outputs/* simulation_outputs/")
    logger.info(f"Results downloaded from VM.")

def upload_results_to_azure(output_dir):
    """
    Uploads all files in the output directory to Azure Blob Storage.

    Parameters:
    output_dir (str): Path to the output directory.
    """
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            local_file_path = os.path.join(root, file)
            blob_name = os.path.relpath(local_file_path, output_dir)
            upload_to_azure(local_file_path, blob_name)

def upload_to_azure(local_file_path, blob_name):
    """
    Uploads a file to Azure Blob Storage.

    Parameters:
    local_file_path (str): Path to the local file.
    blob_name (str): Name of the blob in Azure.
    """
    try:
        blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)

        with open(local_file_path, "rb") as data:
            blob_client.upload_blob(data, overwrite=True)
        logger.info(f"Uploaded {local_file_path} to {blob_name}")
    except Exception as e:
        logger.error(f"Failed to upload {local_file_path} to {blob_name}: {e}")

if __name__ == "__main__":
    # Authenticate with Azure
    credential = DefaultAzureCredential()
    resource_client = ResourceManagementClient(credential, subscription_id)
    network_client = NetworkManagementClient(credential, subscription_id)
    compute_client = ComputeManagementClient(credential, subscription_id)
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

    # Create virtual network, subnet, public IP, and network interface
    create_virtual_network(network_client)
    create_subnet(network_client)
    public_ip = create_public_ip(network_client)
    nic = create_nic(network_client, public_ip)

    # Create virtual machine
    create_vm(compute_client, nic.id)

    # Upload script to VM and run simulation
    upload_script_to_vm(public_ip.ip_address)
    run_simulation_on_vm(public_ip.ip_address)

    # Download results from VM and upload to Azure
    download_results_from_vm(public_ip.ip_address)
    upload_results_to_azure("simulation_outputs")