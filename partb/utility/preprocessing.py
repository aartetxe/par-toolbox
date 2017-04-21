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

    @staticmethod
    def get_fdiag(date_first_diagnostic, date_admission, date_format="%d/%m/%Y"):
        """
        This method calculates the time span between the first diagnostic and the admission, 
        given the admission and first diagnostic dates

        :param date_first_diagnostic: date of the first diagnostic
        :param date_admission: date of the hospital admission
        :param date_format: (optional) format of the date, default='%d/%m/%Y'
        :return: years since first diagnostic
        """

        if pd.isnull(date_first_diagnostic) or pd.isnull(date_admission):
            return "NaN"
        else:
            d0 = datetime.strptime(date_first_diagnostic, date_format)
            d1 = datetime.strptime(date_admission, date_format)
            delta = d1 - d0
            return delta.days / 365

    @staticmethod
    def get_age(day_of_birth, date_admission, date_format="%d/%m/%Y"):
        """
        This method calculates the length of stay (LOS) of an admission, given the
        admission date and the day of birth

        :param day_of_birth: date of patient's day of birth
        :param date_admission: date of the hospital admission
        :param date_format: (optional) format of the date, default='%d/%m/%Y'
        :return: age in years
        """

        if pd.isnull(day_of_birth) or pd.isnull(date_admission):
            return "NaN"
        else:
            d0 = datetime.strptime(day_of_birth, date_format)
            d1 = datetime.strptime(date_admission, date_format)
            delta = d1 - d0
            return delta.days / 365

    def set_los(self, col_admission, col_discharge, date_format="%d/%m/%Y"):
        """
        Creates a new column containing the length of stay (LOS) in days of each admission

        :param col_admission: column name of the hospital admission date
        :param col_discharge: column name of the hospital discharge date
        :param date_format: (optional) format of the date, default='%d/%m/%Y'
        :return: Self
        """

        for i, row in self.df.iterrows():
            los = self.get_los(row[col_admission], row[col_discharge], date_format=date_format)
            self.df.ix[i, 'LOS'] = los

        return self

    def set_first_diagnostic(self, col_first_diagnostic, col_admission, date_format="%d/%m/%Y"):
        """
        Overwrites first diagnostic column with the time since first diagnostic

        :param col_first_diagnostic: column name of the hospital admission date
        :param col_admission: column name of the hospital admission date
        :param date_format: (optional) format of the date, default='%d/%m/%Y'
        :return: Self
        """

        for i, row in self.df.iterrows():
            fdiag = self.get_fdiag(row[col_first_diagnostic], row[col_admission], date_format=date_format)
            self.df.ix[i, col_first_diagnostic] = fdiag

        return self

    def set_age(self, col_day_of_birth, col_admission, date_format="%d/%m/%Y"):
        """
        Creates a new column containing the age of each patient

        :param col_day_of_birth: column name of the day of birth
        :param col_admission: column name of the hospital admission date
        :param date_format: (optional) format of the date, default='%d/%m/%Y'
        :return: Self
        """

        for i, row in self.df.iterrows():
            age = self.get_age(row[col_day_of_birth], row[col_admission], date_format=date_format)
            self.df.ix[i, 'AGE'] = age

        return self
