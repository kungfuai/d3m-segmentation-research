import os
from azure.storage.blob import ContainerClient

# Requires azure-storage-blob==12.3.0


class AzureClient:
    def __init__(self, container=None):
        self.container_name = (
            container if container else os.environ.get("AZURE_CONTAINER_NAME")
        )
        self.client = ContainerClient(
            os.environ.get("AZURE_BLOB_URL"),
            self.container_name,
            credential=os.environ.get("AZURE_SAS_TOKEN"),
        )

    def resource(self):
        return self.client

    def get_file(self, blob_name):
        """Returns file in memory"""
        return self.get_blob(blob_name).download_blob().content_as_bytes()

    def download_file(self, blob_name, local_file_path):
        """Downloads file to local file system."""
        with open(local_file_path, "wb") as fd:
            fd.write(self.get_file(blob_name))

    def put_file(self, blob_name, body):
        """Puts a file in memory into the blob."""
        self.get_blob(blob_name).upload_blob(body)

    def delete_file(self, blob_name, delete_snapshots=False):
        """Deletes file and (optionally) snapshots."""
        self.get_blob(blob_name).delete_blob(delete_snapshots=delete_snapshots)

    def upload_file(self, blob_name, file_name):
        """Upload a file from local file system to blob storage."""
        with open(file_name, "rb") as fd:
            body = fd.read()
        self.get_blob(blob_name).upload_blob(body)

    def list_files(self, filter=None, limit=None):
        """List files in container (note: because of blob structure, this is recursive)."""
        generator = self.client.list_blobs(name_starts_with=filter)
        if limit is None:
            return [blob_object["name"] for blob_object in list(generator)]
        file_list = []
        for blob_object in generator:
            file_list.append(blob_object["name"])
            if len(file_list) == limit:
                return file_list

    def get_blob(self, blob_name):
        """Returns an Azure Blob Client object."""
        return self.client.get_blob_client(blob_name)
