import pandas as pd
import sqlite3
import os

_ROOT = os.path.abspath(os.path.dirname(__file__))
def get_data(path):
    return os.path.join(_ROOT, 'data', path)
DB_NAME = get_data('Stars.sqlite')

class DatabaseReader(object):
    def __init__(self, database_name=DB_NAME):
        self.db_con = sqlite3.connect(database_name)

    def query_object(self, starnames, key='*'):
        """
        Get information about the given star.

        Parameters:
        ===========
        starnames:    string, or iterable of strings
                      The name(s) of the star.

        key:          string, default='*' (return everything)
                      What data do you want? Can be anything that sql will take

        Returns:
        ========
        A pandas DataFrame with the given information for each star
        """

        if isinstance(starnames, str):
            starnames = [starnames,]
        starnames = ["'{}'".format(n) for n in starnames]

        name_list = '(' + ', '.join(starnames) + ')'
        sql_query = "SELECT {} FROM star WHERE name IN {}".format(key, name_list)
        print(sql_query)
        df = pd.read_sql_query(sql_query, self.db_con)

        return df

