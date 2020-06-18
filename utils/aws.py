import warnings
from pathlib import Path
try:
    import boto3
except ModuleNotFoundError:
    warnings.warn('The boto3 library counldn\'t be imported. Ignore if not using AWS s3 buckets', ImportWarning)
    pass


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
