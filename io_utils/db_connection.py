import os
import sqlalchemy as sa
import pandas as pd


class DBConnection(object):
    def __init__(self, config):
        self.db_host_name = config.db_host_name
        self.db_port = config.db_port
        self.db_user = config.db_user
        self.db_password = config.db_password
        self.db_name = config.db_name
        self.db_connection_str = self._db_connection_str()
        self.db_connection = self._connection(self.db_connection_str)

    def _db_connection_str(self):
        db_connection_str = ''.join(str(s) for s in ["mysql+pymysql://",
                                                      self.db_user,
                                                      ':', self.db_password,
                                                      '@', self.db_host_name,
                                                      ':', self.db_port,
                                                      '/', self.db_name])
        return db_connection_str

    def _connection(self, db_connection_str):
        """Define SQL DB Connection"""
        try:
            connection = sa.create_engine(db_connection_str)
        except Exception, e:
            warnings.warn(str(e))
            return None
        return connection



    def pandas_to_mysql(self, df=None, df_chunks=None, table_name='table'):
        if df is None and df_chunks is None:
            warnings.warn("NoneType dataframe supplied")
            return None
        elif df_chunks:
            for chunk in df_chunks:
                chunk.to_sql(name=table_name,
                             if_exists='append', con=con)
        else:
            df.to_sql(name=table_name,
                    if_exists='replace',
                    con=self.db_connection)


if __name__ == '__main__':
    """Unit Test"""

    try:
        db = DBConnection()
        db.pandas_to_mysql(df=None)
        print('output saved as mysql db')
    except Exception, e:
        print(str(e))
