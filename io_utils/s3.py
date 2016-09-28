"""This module has methods to I/O data from aws/S3"""
from __future__ import absolute_import, division, print_function

# Import the SDK
import os
import sys
import boto3
import botocore
import uuid
import pandas as pd
import timeit
import gzip
import ntpath
import warnings
from collections import defaultdict
import timeit



# AWS Credentials
#    1. Environment variables (AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY)
#    2. Credentials file (~/.aws/credentials or
#         C:\Users\USER_NAME\.aws\credentials)
#    3. AWS IAM role for Amazon EC2 instance
#       (http://docs.aws.amazon.com/AWSEC2/latest/UserGuide/iam-roles-for-amazon-ec2.html)





class S3(object):
    """
    A wrapper for boto3 with additional functionality for file transfer

    """
    def __init__(self, profile_name, bucket_name=None):
        if bucket_name is None:
            raise ValueError("No valid bucket name")

        self.session = boto3.Session(profile_name=profile_name)
        test_connection(self.session, bucket_name)

        self.resource = self.session.resource('s3')
        self.client = self.session.client('s3')
        self.bucket = self.resource.Bucket(bucket_name)
        self.bucket_name = bucket_name

    def upload_file(self, upload_file, save_as):
        # Upload the file to S3
        self.client.upload_file(upload_file, self.bucket_name, save_as)

    def download_file(self, download_file, save_as):
        # Download the file from S3
        self.client.download_file(self.bucket.name, download_file, save_as)
        #print(open(save_as).read())

    def upload_dir(self, local_dir, s3_dir):
        for root, dirs, files in os.walk(local_dir):
            # print(dirs)
            exceptions = ['.DS_Store']
            for f in files:
                if f in exceptions:
                    continue
                if not self._is_key_exists(self.bucket_name, s3_dir):
                    try:
                        self.client.put_object(Bucket=self.bucket_name,
                                                Body='',
                                                Key=os.path.join(s3_dir, '/'))
                    except Exception as e:
                        warnings.warn(str(e))
                print(os.path.join(root, f), '\n\t-->', os.path.join(s3_dir, f))
                self.upload_file(os.path.join(root, f), os.path.join(s3_dir, f))

    def download_dir(self, s3_dir, local_dir):
        objs = self.client.list_objects(Bucket=self.bucket_name, Prefix=s3_dir)['Contents']
        objkeys = [o['Key'] for o in objs]

        for k in objkeys:
            file_name = ntpath.basename(k)
            print(os.path.join(k), '\n\t-->', os.path.join(local_dir, file_name))
            save_as = os.path.join(local_dir, file_name)
            try:
                self.download_file(download_file=k, save_as=save_as)
            except Exception, e:
                print(str(e))

    def clean_dir(self, s3_dir):
        """Remove all files from folder

        :s3_dir = ''
        """
        if not self._is_key_exists(self.bucket_name, os.path.join(s3_dir, '/')):
            return None
        try:
            objs = self.client.list_objects(Bucket=self.bucket_name, Prefix=s3_dir)['Contents']
        except Exception as e:
            warnings.warn((e))
            return None
        objkeys = [o['Key'] for o in objs]

        for k in objkeys:
            file_name = ntpath.basename(k)
            if not file_name:
                continue
            print('deleting', k)
            if self._is_key_exists(self.bucket_name, k):
                try:
                    self.client.delete_object(Bucket=self.bucket_name,
                                             Key=k)
                except Exception as e:
                    print(str(e))


    def _is_key_exists(self, bucket_name, key):
        exists = False
        try:
            self.resource.Object(bucket_name, key).load()
        except botocore.exceptions.ClientError as e:
            if e.response['Error']['Code'] == "404":
                exists = False
            else:
                raise e
        else:
            exists = True
        return exists


# Boto 3
def test_connection(session, bucket_name):
    """Test connection and if bucket exists"""
    print('test if bucket', bucket_name, 'exists')
    import botocore
    s3 = session.resource('s3')
    #bucket = s3.Bucket(bucket_name)
    exists = True
    try:
        s3.meta.client.head_bucket(Bucket=bucket_name)
    except botocore.exceptions.ClientError as e:
        # If a client error is thrown, then check that it was a 404 error.
        # If it was a 404 error, then the bucket does not exist.
        error_code = int(e.response['Error']['Code'])
        if error_code == 404:
            exists = False
    print('bucket', bucket_name, 'exists', exists)






def test_list_files(profile_name, bucket_name):
    #bucket_name = 'com.bstis.dev.rsp1.claims'
    session = boto3.Session(profile_name=profile_name)
    s3 = session.resource('s3')
    #s3client = session.client('s3')
    bucket = s3.Bucket(bucket_name)
    test_connection(session, bucket_name)
    print('reading bucket', bucket.name)
    client_list(session, bucket_name)
    # list_files(bucket)






if __name__ == '__main__':
    pass
