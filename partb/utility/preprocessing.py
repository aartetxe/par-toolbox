import pandas as pd
from datetime import datetime


class PreProcessing:
    """
    This class provides some tools for pre-processing EHR data
    """

    def __init__(self,
                 df,
                 target):

        self.df = df
        self.target = target

    @staticmethod
    def get_los(date_admission, date_discharge, date_format="%d/%m/%Y"):
        """
        This method calculates the length of stay (LOS) of an admission, given the
        admission and discharge date

        :param date_admission: date of the hospital admission
        :param date_discharge: date of the hospital discharge
        :param date_format: (optional) format of the date, default='%d/%m/%Y'
        :return: LOS in days
        """

        if pd.isnull(date_admission) or pd.isnull(date_discharge):
            return "NaN"
        else:
            d0 = datetime.strptime(date_admission, date_format)
            d1 = datetime.strptime(date_discharge, date_format)
            delta = d1 - d0
            return delta.days

    def set_los(self, col_admission, col_discharge, date_format="%d/%m/%Y"):
        """
        This method calculates and

        :param col_admission: column name of the hospital admission date
        :param col_discharge: column name of the hospital discharge date
        :param date_format: (optional) format of the date, default='%d/%m/%Y'
        :return: the dataframe with an additional column named 'LOS' containing the LOS in days
        """

        for i, row in self.df.iterrows():
            los = self.get_los(row[col_admission], row[col_discharge], date_format=date_format)
            self.df.ix[i, 'LOS'] = los

        return self
