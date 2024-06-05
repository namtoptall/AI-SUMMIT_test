import boto3
from config import AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_BUCKET_NAME, AWS_REGION

def upload_file_to_s3(file_name, object_name=None):
    s3_client = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY_ID,
                             aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                             region_name=AWS_REGION)
    try:
        s3_client.upload_file(file_name, AWS_BUCKET_NAME, object_name or file_name)
        print(f"File {file_name} uploaded to {AWS_BUCKET_NAME}")
    except Exception as e:
        print(f"Error uploading file: {e}")

def download_file_from_s3(object_name, file_name):
    s3_client = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY_ID,
                             aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                             region_name=AWS_REGION)
    try:
        s3_client.download_file(AWS_BUCKET_NAME, object_name, file_name)
        print(f"File {file_name} downloaded from {AWS_BUCKET_NAME}")
    except Exception as e:
        print(f"Error downloading file: {e}")
