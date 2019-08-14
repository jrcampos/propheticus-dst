#!/usr/bin/env python
"""
Contains the required code to connect to different databases
"""
import pypyodbc
import time
import numpy

import propheticus.shared

class DBI(object):
    DRIVER_MYSQL = "MySQL ODBC 8.0 ANSI Driver"
    DRIVER_SQL_SERVER = "SQL Server"

    MYSQL_PYTHON_DATA_TYPES_MAP = {
        'int': 'int64',
        'float': 'float64',
        'double': 'float',
        'char': 'string'
    }

    """
    Contains the required code to connect to different databases

    Parameters
    ----------
    driver
    server
    user
    password
    database
    """
    def __init__(self, driver, server, user, password, database):
        self.driver = driver
        self.server = server
        self.user = user
        self.password = password
        self.database = database
        self.connect()

    def connect(self):
        """
        Creates the connection to the database

        Returns
        -------

        """
        try:
            self.connection_string = ";".join([
                'Driver={' + self.driver + '}',
                'Server=' + self.server,
                'Database=' + self.database,
                'uid=' + self.user,
                'pwd=' + self.password
            ])
            self.connection = pypyodbc.connect(self.connection_string)

        except ValueError:
            exit('An error occurred when connecting to the database: ' + ValueError)

    def query(self, SQL):
        """
        Performs a given SQL query

        Parameters
        ----------
        SQL : str

        Returns
        -------
        Data : object

        """
        retry_count_max = 5
        retry_count = 0
        retry_flag = True
        Data = None
        while retry_flag and retry_count < retry_count_max:
            try:
                cursor = self.connection.cursor()
                cursor.execute(SQL)
                Data = cursor.fetchall()
                cursor.close()
                retry_flag = False

            except:
                self.reconnect()
                propheticus.shared.Utils.printErrorMessage(self.database + ': An error occurred when querying to the database: ' + SQL)
                retry_count = retry_count + 1
                time.sleep(1)

        if retry_count == 5:
            propheticus.shared.Utils.printErrorMessage('Query could not be executed after ' + str(retry_count_max) + ' tries: ' + SQL)

        return Data

    def reconnect(self):
        """
        Reconnects to the database

        Returns
        -------

        """
        self.close()
        self.connect()

    def close(self):
        """
        Closes the connection with the database

        Returns
        -------

        """
        self.connection.close()
