import boto3
import logging
from botocore.config import Config
from botocore.exceptions import ClientError
import time, datetime
from string import Template
import warnings
from pathlib import Path


def download_s3_files(bucket_name, data_path, output_path):
    """
    Function to download the required training files from s3 bucket and sets ec2 paths.
    :param bucket_name: (str) bucket in which data is stored if using AWS S3
    :param data_path: (str) EC2 file path of the folder containing h5py files
    :param output_path: (str) EC2 file path in which the model will be saved
    :return: (S3 object) bucket, (str) bucket_output_path, (str) local_output_path, (str) data_path
    """
    bucket_output_path = output_path
    local_output_path = Path('output_path')
    output_path.mkdir(exist_ok=True)
    s3 = boto3.resource('s3')
    bucket = s3.Bucket(bucket_name)

    if data_path:
        bucket.download_file(data_path.joinpath('samples/trn_samples.hdf5'), 'samples/trn_samples.hdf5')
        bucket.download_file(data_path.joinpath('samples/val_samples.hdf5'), 'samples/val_samples.hdf5')
        bucket.download_file(data_path.joinpath('samples/tst_samples.hdf5'), 'samples/tst_samples.hdf5')

    return bucket, bucket_output_path, local_output_path, data_path

def write_to_s3(bucket_name, region_name, aws_access_key_id, aws_secret_access_key, local_filename, bucket_filename, mime_type):
    '''
    Given all access information to an S3 bucket (params below), copy 'local_filename' to 'bucket_filename'.
    By default the file written to S3 'expires' after 1 day; WARNING this does not erase the file from the bucket
    :param bucket_name: (str) ex. 'datacube-dev-data-scratch'
    :param region_name: (str) AWS region name, e.g. 'ca-central-1'
    :param aws_access_key_id: (str)
    :param aws_secret_access_key: (str)
    :param local_filename: (str) path to the local file to be transferred to S3
    :param bucket_filename: (str) desired filename as it would appear on an S3 url; see 'destination_url' below
    :param mime_type: (str) ex. 'image/geo+tiff' for a COG or 'application/geopackage+vnd.sqlite3' for a GeoPackage
    :return: destination_url, a concatenation of bucket_name + region_name + bucket_filename
    WARNING : There is no provision for filename collisions e.g. if two users write different files to the same url
    '''
    destination_url_template = Template('https://$S3_BUCKET_NAME.s3.$S3_REGION.amazonaws.com/$S3_KEY')
    destination_url = destination_url_template.substitute(S3_BUCKET_NAME=bucket_name, S3_REGION=region_name, S3_KEY=bucket_filename)

    utc_now = datetime.datetime.utcnow()
    expires = utc_now + datetime.timedelta(days=1)
    expires_unixtime = time.mktime(expires.timetuple())

    s3_client = boto3.client(
        's3',
        aws_access_key_id = aws_access_key_id,
        aws_secret_access_key = aws_secret_access_key)

    try:
        response = s3_client.upload_file(local_filename, bucket_name, bucket_filename,
                                         ExtraArgs={'ACL': 'public-read', 'Expires':str(expires_unixtime), 'ContentType':mime_type})
        data_object = s3_client.get_object(Bucket=bucket_name, Key=bucket_filename)
        valid_till = data_object['ResponseMetadata']['HTTPHeaders']['expires']
        mime_type = data_object['ContentType']
        print("\nInference results available at\n" + destination_url + "\nas MIME type " + mime_type + "\nuntil " + valid_till)
        return destination_url

    except ClientError as e:
        logging.error(e)
        print("error")
