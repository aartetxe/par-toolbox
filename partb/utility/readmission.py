import pandas as pd
from datetime import timedelta


class Readmission:
    """
    This class provides some tools for extracting readmission-related features
    """
    def __init__(self, df):

        self.df = df
        self.id = None
        self.col_admission = None
        self.col_discharge = None
        self.date_format = None
        self.grouped = None

    def prepare(self, patient_id, col_admission, col_discharge, date_format="%d/%m/%Y"):
        """
        Performs some data formatting and grouping for further processing

        :param patient_id: column name of the patient id
        :param col_admission: column name of the admission date
        :param col_discharge: column name of the discharge date
        :param date_format: (optional) format of the date, default='%d/%m/%Y'
        :return:
        """
        # Set values
        self.id = patient_id
        self.col_admission = col_admission
        self.col_discharge = col_discharge
        self.date_format = date_format
        # Convert string dates into datetime
        self.df[col_admission] = pd.to_datetime(self.df[col_admission], format=date_format)
        self.df[col_discharge] = pd.to_datetime(self.df[col_discharge], format=date_format)
        # Group by patient
        self.grouped = self.df.groupby(patient_id)

        return self

    def admission_days(self):
        """
        Creates a new column containing the days passed between a discharge and the following admission

        :return: Self
        """

        # TODO: Check data is already prepared
        for idx, g in self.grouped:  # Iterate through every patient
            sortedf = g.sort_values(by=self.col_admission)  # Sort by admission date
            oldi = None
            for i, row in sortedf.iterrows():  # Iterate admissions
                if oldi is None:  # Is first admission
                    oldi = i
                else:
                    date_discharge = sortedf.ix[oldi][self.col_discharge]
                    date_admission = row[self.col_admission]
                    # Check there are not null values
                    if pd.isnull(date_admission) or pd.isnull(date_discharge):
                        continue
                    else:
                        delta = date_admission - date_discharge
                        self.df.ix[oldi, 'readmission_days'] = delta.days

                    oldi = i

        return self

    def previous_admissions(self, threshold=180):
        """
        Creates a new column containing the number of previous admissions given a threshold

        :param threshold: (optional) number of days to count from, default=180
        :return: Self
        """

        # TODO: Check data is already prepared
        for idx, g in self.grouped:  # Iterate through every patient
            sortedf = g.sort_values(by=self.col_admission)  # Sort by admission date
            for i, row in sortedf.iterrows():  # Iterate admissions
                date_discharge = row[self.col_admission]
                date_threshold = date_discharge - timedelta(days=threshold)

                prevs = sortedf[(sortedf[self.col_admission] < date_discharge) & (date_threshold < sortedf[self.col_admission])]

                self.df.ix[i, 'previous_admissions'] = len(prevs[self.col_admission])

        return self
