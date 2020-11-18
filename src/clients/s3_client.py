from os import environ
from boto3 import resource


class S3Client(object):
    def __init__(self, bucket=None):
        self.access_key_id = environ.get("AWS_S3_ACCESS_KEY_ID")
        self.secret_access_key = environ.get("AWS_S3_SECRET_ACCESS_KEY")
        self.bucket_name = bucket if bucket else environ.get("AWS_S3_BUCKET_NAME")
        self.s3 = resource(
            "s3",
            aws_access_key_id=self.access_key_id,
            aws_secret_access_key=self.secret_access_key,
        )

    def resource(self):
        return self.s3

    def get_file(self, key):
        """Returns file in memory"""
        return self.s3.Object(self.bucket_name, key).get()["Body"].read()

    def download_file(self, key, local_file_path):
        """Downloads file at "key" to local file system"""
        self.s3.Bucket(self.bucket_name).download_file(key, local_file_path)

    def put_file(self, key, body):
        """Puts a file in memory onto the s3 bucket"""
        self.s3.Object(self.bucket_name, key).put(Body=body)

    def upload_file(self, key, file_name):
        """Uploads a file from local file system to s3 bucket"""
        self.s3.Bucket(self.bucket_name).upload_file(file_name, key)

    def delete_file(self, key):
        self.s3.Object(self.bucket_name, key).delete()

    def list_files(self, filter=None, limit=None):
        objects = self.s3.Bucket(self.bucket_name).objects
        objects = objects.filter(Prefix=filter) if filter else objects
        objects = objects.limit(limit) if limit else objects
        objects = objects.all() if limit is None and filter is None else objects
        return [object.key for object in objects]

    def move_file(self, current_key, new_key):
        # Not used
        self.s3.Object(self.bucket_name, new_key).copy_from(
            CopySource={"Bucket": self.bucket_name, "Key": current_key}
        )
        self.s3.Object(self.bucket_name, current_key).delete()
