from os import environ
from google.cloud import storage
from json import dumps


class GCSClient:
    def __init__(self, bucket=None, project=None):
        self.bucket_name = bucket if bucket else environ.get("GCS_BUCKET_NAME")
        self.gcs = storage.Client.from_service_account_json("config/gcp_secrets.json")

    def resource(self):
        return self.gcs

    def get_file(self, blob_name):
        """Returns file in memory"""
        return self.get_blob(blob_name).download_as_string()

    def download_file(self, blob_name, local_file_path):
        """Downloads file at "blob_name" to local file system"""
        self.get_blob(blob_name).download_to_filename(local_file_path)

    def put_file(self, blob_name, body):
        """Puts a file in memory onto the gcs bucket"""
        self.get_blob(blob_name).upload_from_string(body)

    def upload_file(self, blob_name, file_name):
        """Uploads a file from local file system to gcs bucket"""
        self.get_blob(blob_name).upload_from_filename(file_name)

    def delete_file(self, blob_name):
        self.get_blob(blob_name).delete()

    def list_files(self, filter=None, limit=None):
        bucket = self.get_bucket()
        blobs = self.gcs.list_blobs(bucket, prefix=filter, max_results=limit)
        return [blob.name for blob in blobs]

    def get_bucket(self):
        return self.gcs.get_bucket(self.bucket_name)

    def get_blob(self, blob_name):
        return self.get_bucket().blob(blob_name)
