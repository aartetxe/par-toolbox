from setuptools import setup

NAME = "partb"
VERSION = "0.4.0"
DESCRIPTION = "Predictive Analytics for Readmission: a Toolbox in Python"
KEYWORDS = "Predictive Data Analytics Readmission Toolbox"
AUTHOR = "Arkaitz Artetxe"
AUTHOR_EMAIL = "aartetxe@vicomtech.org"
URL = "XXX"

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    keywords=KEYWORDS,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    url=URL,
    packages=['partb', 'partb.classification', 'partb.visualization', 'partb.descriptive', 'partb.utility'],
    install_requires=['numpy', 'sklearn', 'scipy', 'matplotlib', 'seaborn', 'pandas', 'imblearn']
)
