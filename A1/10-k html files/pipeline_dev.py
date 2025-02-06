""" Pipeline Development
"""

# S1: iterate over all the 10-k files in the directory
import re
from pathlib import Path
import pandas as pd
from datetime import datetime
import html

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
    '10-K Legal Proceedings. The Company is involved in various legal proceedings arising in the ordinary course of business.'

    >>> with open('AAPL_10K.html') as file:
    ...     test_apple = file.read()[0:1000]
    >>> clean_html(test_apple)
    'aapl-20240928'

    >>> JnJ = " For the fiscal year ended <ix:nonNumeric contextRef='c-1' name='dei:DocumentPeriodEndDate' format='ixt:date-monthname-day-year-en' id='f-4'><ix:nonNumeric contextRef='c-1' name='dei:CurrentFiscalYearEndDate' format='ixt:date-monthname-day-en' id='f-3'>December 31</ix:nonNumeric>, 2023"
    >>> clean_html(JnJ)
    'For the fiscal year ended December 31 , 2023'

    >>> GOOG = "For the fiscal year ended <ix:nonNumeric contextRef='c-1' name='dei:DocumentPeriodEndDate' format='ixt:date-monthname-day-year-en' id='f-4'><ix:nonNumeric contextRef='c-1' name='dei:CurrentFiscalYearEndDate' format='ixt:date-monthname-day-en' id='f-3'>December&#160;31</ix:nonNumeric>, 2023"
    >>> clean_html(GOOG)
    'For the fiscal year ended December 31 , 2023'

    >>> AAPLS = '''SIGNATURES</span></div><div style='margin-top:12pt;text-align:justify'><span style='color:#000000;font-family:'Helvetica',sans-serif;font-size:9pt;font-weight:400;line-height:120%'>Pursuant to the requirements of Section 13 or 15(d) of the Securities Exchange Act of 1934, the Registrant has duly caused this report to be signed on its behalf by the undersigned, thereunto duly authorized.</span></div><div style='margin-top:6pt;text-align:justify'><table style='border-collapse:collapse;display:inline-table;margin-bottom:5pt;vertical-align:text-bottom;width:100.000%'><tr><td style='width:1.0%'/><td style='width:56.794%'/><td style='width:0.1%'/><td style='width:1.0%'/><td style='width:2.847%'/><td style='width:0.1%'/><td style='width:0.1%'/><td style='width:2.139%'/><td style='width:0.1%'/><td style='width:1.0%'/><td style='width:34.720%'/><td style='width:0.1%'/></tr><tr><td colspan="3" style='padding:2px 1pt;text-align:left;vertical-align:bottom'><div><span style='color:#000000;font-family:'Helvetica',sans-serif;font-size:9pt;font-weight:400;line-height:100%'>Date: November&#160;1, 2024</span>'''
    >>> clean_html(AAPLS)
    'SIGNATURES Pursuant to the requirements of Section 13 or 15(d) of the Securities Exchange Act of 1934, the Registrant has duly caused this report to be signed on its behalf by the undersigned, thereunto duly authorized. Date: November 1, 2024'

    >>> Singers = '''<span style="color:#000000;font-family:'Helvetica',sans-serif;font-size:9pt;font-weight:400;line-height:100%">By:</span></td><td colspan="3" style="padding:0 1pt"/><td colspan="3" style="padding:2px 1pt;text-align:left;vertical-align:bottom"><span style="color:#000000;font-family:'Helvetica',sans-serif;font-size:9pt;font-weight:400;line-height:100%">/s/ Luca Maestri</span>'''
    >>> clean_html(Singers)
    'By: /s/ Luca Maestri'

    >>> AWS = '''SIGNATURES</span></div><div style="margin-top:5pt;text-indent:24.75pt"><span style="color:#000000;font-family:'Times New Roman',sans-serif;font-size:10pt;font-weight:400;line-height:120%">Pursuant to the requirements of Section&#160;13 or 15(d) of the Securities Exchange Act of 1934, the registrant has duly caused this Report to be signed on its behalf by the undersigned, thereunto duly authorized, as of February&#160;1, 2024. </span>'''
    >>> clean_html(AWS)
    'SIGNATURES Pursuant to the requirements of Section 13 or 15(d) of the Securities Exchange Act of 1934, the registrant has duly caused this Report to be signed on its behalf by the undersigned, thereunto duly authorized, as of February 1, 2024.'
    >>> APPL2 = '''<span style="color:#000000;font-family:'Helvetica',sans-serif;font-size:9pt;font-weight:700;line-height:120%">SIGNATURES</span></div><div style="margin-top:12pt;text-align:justify"><span style="color:#000000;font-family:'Helvetica',sans-serif;font-size:9pt;font-weight:400;line-height:120%">Pursuant to the requirements of Section 13 or 15(d) of the Securities Exchange Act of 1934, the Registrant has duly caused this report to be signed on its behalf by the undersigned, thereunto duly authorized.</span></div><div style="margin-top:6pt;text-align:justify"><table style="border-collapse:collapse;display:inline-table;margin-bottom:5pt;vertical-align:text-bottom;width:100.000%"><tr><td style="width:1.0%"/><td style="width:56.794%"/><td style="width:0.1%"/><td style="width:1.0%"/><td style="width:2.847%"/><td style="width:0.1%"/><td style="width:0.1%"/><td style="width:2.139%"/><td style="width:0.1%"/><td style="width:1.0%"/><td style="width:34.720%"/><td style="width:0.1%"/></tr><tr><td colspan="3" style="padding:2px 1pt;text-align:left;vertical-align:bottom"><div><span style="color:#000000;font-family:'Helvetica',sans-serif;font-size:9pt;font-weight:400;line-height:100%">Date: November&#160;1, 2024</span>'''
    >>> clean_html(APPL2)
    'SIGNATURES Pursuant to the requirements of Section 13 or 15(d) of the Securities Exchange Act of 1934, the Registrant has duly caused this report to be signed on its behalf by the undersigned, thereunto duly authorized. Date: November 1, 2024'
    """
    # remove html tags
    clean_one = re.sub(r'<.*?>', ' ', raw_html)
    # remove html entities
    clean_two = re.sub(r'&\w+;', ' ', clean_one)
    clean_two = re.sub(r'&nbsp;', ' ', clean_two)
    clean_two = re.sub(r'&amp;', '&', clean_two)
    # 去除所有 HTML 实体（如 &#160;）
    clean_two = re.sub(r'&#\d+;', ' ', clean_two)

    # clean_two = html.unescape(clean_two) 去除所有 HTML 实体（如 &nbsp;）的包

    clean_three = re.sub(r"&[a-z]+;", " ", clean_two)

    # remove urls
    clean_three = re.sub(r"\(http[s]?://\S+\)", "", clean_three)
    clean_three = re.sub(r"http[s]?://\S+", "", clean_three)
    # remove multiple spaces
    clean_four = re.sub(r'\s+', ' ', clean_three)
    return clean_four.strip()

with open('AAPL_10K.html') as file:
    raw_content = file.read()
    print(clean_html(raw_content))


def convert_date(date_str):
    """ This is a helper function that converts date to ISO format """
    date = datetime.strptime(date_str, "%B %d, %Y").date()
    return date.isoformat()

def extract_signatures(text):
    """
    This function is used to extract the signers from the text
    >>> test = ('<html xmlns="http://www.w3.org/1999/xhtml"> <head>'
    ...         '<title>10-K</title> <meta http-equiv="Content-Type" content="text/html; charset=utf-8"/>'
    ...         '</head> <body> <div style="font-family:Times New Roman;font-size:10pt;">'
    ...         'Signature Date: January 31, 2024 /s/ John Doe John Doe Chief Executive Officer')
    >>> extract_signatures(test)
    ('John Doe')
    """
    # find the signature section
    signature_section_pattern = r"(SIGNATURES|Signatures)\s"
    pass

def extract_fiscal_year(text):
    """ 1. The date of the fiscal year-end (ensure it is formatted in ISO-format)

    Currently, relevant information is in the beginning of the document, e.g.:
     'For the fiscal year ended January 31, 2024, or'
     Search for the fiscal year and return it in ISO-format.

     >>> extract_fiscal_year('For the fiscal year ended January 31, 2024, or')
     '2024-01-31'
     >>>
     """
    match = re.search(r"for the fiscal year ended (\w+\s\d{1,2},\s\d{4})", text, re.IGNORECASE)
    if match:
        fiscal_year_str = match.group(1)
        fiscal_year_iso = convert_date(fiscal_year_str)
    else:
        fiscal_year_iso = 'N/A'
    return fiscal_year_iso