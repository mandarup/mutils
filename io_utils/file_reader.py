from __future__ import absolute_import, division, print_function

import gzip
import pandas as pd
import s3fs
import os
import numpy as np
import timeit

class CSVReaderS3:
    def __init__(self, profile_name):
        self.profile_name = profile_name

        self.s3f = s3fs.S3FileSystem(anon=False,
                          profile_name=profile_name)

    def read_csv(self, bucket_name, file_name, usecols=None):
        filepath = os.path.join(bucket_name, file_name)
        with self.s3f.open(filepath, 'rb') as f:
            gz = gzip.GzipFile(fileobj=f)  # Decompress data with gzip
            df = pd.read_csv(gz,
                             index_col=False,
                             usecols=usecols,
                             compression='infer')
        return df



class CSVReaderLocal:
    def __init__(self):
        pass

    def read_csv(file_name, usecols=None):
        data = pd.read_csv(file_name, index_col=False, usecols=usecols, compression='gzip')
        return data



def file_reader(config):
    """
    Conditional function definitions based on input data location at s3/local
    """
    if config.read_from_location == 'aws-s3':
        return CSVReaderS3(config)
    elif config.read_from_location == 'local':
        return CSVReaderLocal(config)


if __name__ == "__main__":
    pass
