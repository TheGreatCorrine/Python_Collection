""" Pipeline Development
"""

# S1: iterate over all the 10-k files in the directory
import re
from pathlib import Path
import pandas as pd
from nbconvert.filters import clean_html

"""
Step 1: Iterate over all the 10-k files in the directory
"""
current_dictionary = Path.cwd()
# TODO: Path.cwd() or Path('A1/10-k html files')
for html_file in current_dictionary.glob('*.html'):
    # TODO: utf-8 encoding or ISO-8859-1 encoding or ASCII?
    with html_file.open('r', encoding='utf-8') as file:
        # print(file.read()[0:100])
        # read or readlines? Which one is more space efficient?
        raw_content = file.read()
        # print(raw_content[0:100])
        # print('below are cleaned content')
        # print(clean_html(raw_content)[0:100])



"""
Step 2: Extract all necessary information from the 10-k files
"""
def clean_html(raw_html):
    """
    This function is used to clean the raw content
    >>> test = ('<html xmlns="http://www.w3.org/1999/xhtml"> <head>'
    ...         '<title>10-K</title> <meta http-equiv="Content-Type" content="text/html; charset=utf-8"/>'
    ...         '</head> <body> <div style="font-family:Times New Roman;font-size:10pt;">'
    ...         'Legal Proceedings. The Company is involved in various legal proceedings arising in the ordinary course of business. ')
    >>> clean_html(test)
    'Legal Proceedings. The Company is involved in various legal proceedings arising in the ordinary course of business.'
    """
    # TODO: should 10-k be removed? what is the definition of html tags?
    # remove html tags
    clean_one = re.sub(r'<.*?>', ' ', raw_html)
    # remove html entities
    clean_two = re.sub(r'&\w+;', ' ', clean_one)

    clean_three = re.sub(r"&[a-z]+;", " ", clean_two)

    # remove multiple spaces
    clean_four = re.sub(r'\s+', ' ', clean_three)
    return clean_four.strip()

with open('AAPL-10K.html') as file:
    raw_content = file.read()
    print(clean_html(raw_content)[0:100])