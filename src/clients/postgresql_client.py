import os

import pandas as pd
import sqlalchemy
from sqlalchemy import create_engine


class PostgresqlClient:
    def __init__(self, database=None, host=None, username=None, password=None):
        """Initialize new database client."""
        self.database = database
        self.host = host
        self.username = username
        self.password = password
        self.connection = None
        self.engine = None
        self._get_environment()

    def _get_environment(self):
        """Get environment if database connection parameters weren't specified."""
        if self.database is None:
            self.database = os.environ.get("APP_DATABASE_NAME")
        if self.host is None:
            self.host = os.environ.get("APP_DATABASE_HOST")
        if self.username is None:
            self.username = os.environ.get("APP_DATABASE_USERNAME")
        if self.password is None:
            self.password = os.environ.get("APP_DATABASE_PASSWORD")

    def connect(self):
        """Connect to the database given the specified connection parameters."""
        if self.engine is None:
            try:
                self.engine = create_engine(
                    "postgresql+psycopg2://{}:{}@{}:{}/{}".format(
                        self.username,
                        self.password,
                        self.host,
                        5432,
                        self.database,
                        pool_size=10,
                        max_overflow=0,
                    )
                )
                self.connection = self.engine.connect()
            except sqlalchemy.exc.OperationalError:
                raise Exception(
                    "Could not connect to database given the current environment."
                )

    def close(self):
        """Close an existing connection."""
        if self.connection is not None:
            try:
                self.connection.close()
                self.engine.dispose()
            except:
                pass
        self.engine = None
        self.connection = None

    def read_sql(self, query, parse_dates=None):
        """Read SQL query into a pandas dataframe from the remote database, while parsing specified columns as dates.."""
        if self.engine is None:
            self.connect()
        return pd.read_sql(query, self.engine, parse_dates=parse_dates)
