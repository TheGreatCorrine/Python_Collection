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
def clean_html(raw_html, remove_tags=True):
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

    >>> WMT = '''<span style="color:#000000;font-family:'Times New Roman',sans-serif;font-size:10pt;font-weight:400;line-height:100%">/s/ C. Douglas McMillon</span></td></tr><tr><td colspan="3" style="padding:0 1pt"/><td colspan="3" style="padding:2px 1pt;text-align:left;vertical-align:bottom"><span style="color:#000000;font-family:'Times New Roman',sans-serif;font-size:8pt;font-weight:400;line-height:100%">&#160;</span></td><td colspan="3" style="padding:0 1pt"/><td colspan="3" style="padding:2px 1pt;text-align:left;vertical-align:bottom"><span style="color:#000000;font-family:'Times New Roman',sans-serif;font-size:8pt;font-weight:400;line-height:100%">&#160;</span></td><td colspan="3" style="border-top:1pt solid #000000;padding:2px 1pt;text-align:left;vertical-align:bottom"><span style="color:#000000;font-family:'Times New Roman',sans-serif;font-size:10pt;font-weight:400;line-height:100%">C. Douglas McMillon</span></td></tr><tr><td colspan="3" style="padding:0 1pt"/><td colspan="3" style="padding:2px 1pt;text-align:left;vertical-align:bottom"><span style="color:#000000;font-family:'Times New Roman',sans-serif;font-size:8pt;font-weight:400;line-height:100%">&#160;</span></td><td colspan="3" style="padding:0 1pt"/><td colspan="3" style="padding:2px 1pt;text-align:left;vertical-align:bottom"><span style="color:#000000;font-family:'Times New Roman',sans-serif;font-size:8pt;font-weight:400;line-height:100%">&#160;</span></td><td colspan="3" style="padding:2px 1pt;text-align:left;vertical-align:bottom"><span style="color:#000000;font-family:'Times New Roman',sans-serif;font-size:10pt;font-weight:400;line-height:100%">President and Chief Executive Officer</span></td></tr></table>'''
    >>> clean_html(WMT)
    '/s/ C. Douglas McMillon C. Douglas McMillon President and Chief Executive Officer'

    >>> GOOGL = '''<span style="color:#000000;font-family:'Arial',sans-serif;font-size:10pt;font-weight:700;line-height:120%">SIGNATURES</span></div><div style="margin-top:9pt;text-align:justify;text-indent:24.75pt"><span style="color:#000000;font-family:'Arial',sans-serif;font-size:10pt;font-weight:400;line-height:120%">Pursuant to the requirements of Section&#160;13 or 15(d) of the Securities Exchange Act of 1934, the Registrant has duly caused this Annual Report on Form 10-K to be signed on its behalf by the undersigned, thereunto duly authorized.</span></div><div style="margin-top:9pt;text-align:justify;text-indent:22.5pt"><span style="color:#000000;font-family:'Arial',sans-serif;font-size:10pt;font-weight:400;line-height:120%">Date: February&#160;4, 2025 </span></div><div style="margin-top:6pt;text-align:justify;text-indent:22.5pt"><span style="color:#000000;font-family:'Arial',sans-serif;font-size:9pt;font-weight:400;line-height:120%">&#160;</span><table style="border-collapse:collapse;display:inline-table;margin-bottom:5pt;vertical-align:text-bottom;width:42.522%"><tr><td style="width:1.0%"/><td style="width:16.685%"/><td style="width:0.1%"/><td style="width:1.0%"/><td style="width:81.115%"/><td style="width:0.1%"/></tr><tr><td colspan="6" style="padding:2px 1pt;text-align:left;vertical-align:bottom"><span style="color:#000000;font-family:'Arial',sans-serif;font-size:10pt;font-weight:700;line-height:100%">ALPHABET INC.</span></td></tr><tr><td colspan="3" style="padding:2px 1pt;text-align:left;vertical-align:top"><span style="color:#000000;font-family:'Arial',sans-serif;font-size:10pt;font-weight:400;line-height:100%">By:</span></td><td colspan="3" style="padding:2px 1pt;text-align:left;vertical-align:bottom"><div style="text-align:center"><span style="color:#000000;font-family:'Arial',sans-serif;font-size:10pt;font-weight:400;line-height:100%">/</span><span style="color:#000000;font-family:'Arial',sans-serif;font-size:8pt;font-weight:400;line-height:100%">S</span><span style="color:#000000;font-family:'Arial',sans-serif;font-size:10pt;font-weight:400;line-height:100%">/&#160;&#160;&#160;&#160;S</span><span style="color:#000000;font-family:'Arial',sans-serif;font-size:8pt;font-weight:400;line-height:100%">UNDAR</span><span style="color:#000000;font-family:'Arial',sans-serif;font-size:10pt;font-weight:400;line-height:100%"> P</span><span style="color:#000000;font-family:'Arial',sans-serif;font-size:8pt;font-weight:400;line-height:100%">ICHAI&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;</span></div></td></tr><tr><td colspan="3" style="padding:0 1pt"/><td colspan="3" style="border-top:1pt solid #000000;padding:2px 1pt;text-align:center;vertical-align:bottom"><span style="color:#000000;font-family:'Arial',sans-serif;font-size:8pt;font-weight:400;line-height:100%">Sundar Pichai</span></td></tr><tr><td colspan="3" style="padding:0 1pt"/><td colspan="3" style="padding:2px 1pt;text-align:center;vertical-align:bottom"><span style="color:#000000;font-family:'Arial',sans-serif;font-size:8pt;font-weight:400;line-height:100%">Chief Executive Officer<br/>(Principal Executive Officer of the Registrant)</span></td></tr></table></div>'''
    >>> clean_html(GOOGL)
    ' SIGNATURES Pursuant to the requirements of Section 13 or 15(d) of the Securities Exchange Act of 1934, the Registrant has duly caused this Annual Report on Form 10-K to be signed on its behalf by the undersigned, thereunto duly authorized. Date: February 4, 2025 ALPHABET INC. By: /s/ SUNDAR PICHAI Sundar Pichai Chief Executive Officer (Principal Executive Officer of the Registrant)'

    >>> clean_html(GOOGL, remove_tags=False)

    >>> MSFT = ''' <p style="font-size:10pt;margin-top:0;font-family:Times New Roman;margin-bottom:0;text-align:center;" id="signatures"><span style="color:#000000;white-space:pre-wrap;font-weight:bold;font-size:10pt;font-family:Arial;min-width:fit-content;">SIGNAT</span><span style="color:#000000;white-space:pre-wrap;font-weight:bold;font-size:10pt;font-family:Arial;min-width:fit-content;">URES</span></p> <p style="font-size:10pt;margin-top:9pt;font-family:Times New Roman;margin-bottom:0;text-align:justify;"><span style="color:#000000;white-space:pre-wrap;font-size:10pt;font-family:Arial;min-width:fit-content;">Pursuant to the requirements of Section 13 or 15(d) of the Securities Exchange Act of 1934, the Registrant has duly caused this report to be signed on its behalf by the undersigned; thereunto duly authorized, in the City of Redmond, State of Washington, on July 30, 2024.</span><span style="color:#000000;white-space:pre-wrap;font-size:10pt;font-family:Arial;min-width:fit-content;"> </span></p>'''
    >>> clean_html(MSFT)

    >>> clean_html(LONG_MSFT)
    ' SIGNATURES Pursuant to the requirements of Section 13 or 15(d) of the Securities Exchange Act of 1934, the Registrant has duly caused this report to be signed on its behalf by the undersigned; thereunto duly authorized, in the City of Redmond, State of Washington, on July 30, 2024. MICROSOFT CORPORATION /s/ ALICE L. JOLLA'

    >>> clean_html(TSLA, remove_tags=False)


    """
    # # remove html tags
    # clean_one = re.sub(r'<.*?>', ' ', raw_html)
    # # remove html entities
    # clean_two = re.sub(r'&\w+;', ' ', clean_one)
    # clean_two = re.sub(r'&nbsp;', ' ', clean_two)
    # clean_two = re.sub(r'&amp;', '&', clean_two)
    # # 去除所有 HTML 实体（如 &#160;）
    # clean_two = re.sub(r'&#\d+;', ' ', clean_two)
    #
    # # clean_two = html.unescape(clean_two) 去除所有 HTML 实体（如 &nbsp;）的包
    #
    # clean_three = re.sub(r"&[a-z]+;", " ", clean_two)
    #
    # # remove urls
    # clean_three = re.sub(r"\(http[s]?://\S+\)", "", clean_three)
    # clean_three = re.sub(r"http[s]?://\S+", "", clean_three)
    # # remove multiple spaces
    # clean_four = re.sub(r'\s+', ' ', clean_three)
    # return clean_four.strip()
    if remove_tags:
        # remove HTML tags
        cleaned_html = re.sub(r'<.*?>', ' ', raw_html)
        # remove HTML entities
        cleaned_html = re.sub(r'&\w+;', ' ', cleaned_html)
        cleaned_html = re.sub(r"&[a-z]+;", " ", cleaned_html)

    else:
        cleaned_html = re.sub(r'(<br\s*/?>|</div>|</p>|</tr>|</li>|</table>|</td>)', '\n', raw_html, flags=re.IGNORECASE)
        cleaned_html = re.sub(r'<(?!/)[^>]+>', '', cleaned_html, flags=re.IGNORECASE) # do not remove <\span>
        # Remove <span> tags that are immediately between non-whitespace characters
        cleaned_html = re.sub(r'(?<=\S)<span>(?=\S)', '', cleaned_html)

        # Remove </span> tags that are immediately between non-whitespace characters
        cleaned_html = re.sub(r'(?<=\S)</span>(?=\s)', '', cleaned_html)

        cleaned_html = re.sub(r'(?<=\S)</span>(?=\S)', '', cleaned_html)
        # cleaned_html = re.sub(r'(?<=\S)</span>(?=\s(?!\S))', '', cleaned_html)

    # replace HTML entities (&#160;）with space
    cleaned_html = re.sub(r'&#\d+;|nbsp', ' ', cleaned_html)
    cleaned_html = re.sub(r'\s+', ' ', cleaned_html)

    # remove urls
    cleaned_url = re.sub(r"\(http[s]?://\S+\)", "", cleaned_html)
    cleaned_url = re.sub(r"http[s]?://\S+", "", cleaned_url)

    # remove multiple spaces
    cleaned_space = re.sub(r'\s+', ' ', cleaned_url)

    # fix the error that /s/ is splitted into / s / in GOOG_10-K_2021.html
    clean_spe = re.sub(r'/\s*S\s*/', '/s/', cleaned_space, flags=re.IGNORECASE)

    # fix the error: ALICE is splitted into multiple A LICE in MSFT_10-K_2021.html
    clean_spe = re.sub(r'(\b[A-Z])\s([A-Z]{2,}\b)', r'\1\2', clean_spe)

    # fix the error: can not find the signature section in MSFT_10-K_2021.html
    cleaned = re.sub(r'SIGNAT\s*URES', 'SIGNATURES', clean_spe, flags=re.IGNORECASE)

    return cleaned.strip()



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

def extract_signature_date(text):
    """ 3. The date of signature(s) (ensure it is formatted in ISO-format).

    Find the SIGNATURES section
    >>> test = 'SIGNATURES Pursuant to the requirements of Section 13 or 15(d) of the Securities Exchange Act of 1934, the Registrant has duly caused this report to be signed on its behalf by the undersigned, thereunto duly authorized. Date: November 1, 2024'
    >>> extract_signature_date(test)
    '2024-11-01'
    >>> test_duplicates = 'SIGNATURES Pursuant to the requirements of Section 13 or 15(d) of the Securities Exchange Act of 1934, the Registrant has duly caused this report to be signed on its behalf by the undersigned, thereunto duly authorized. Date: November 1, 2024 Date: November 2, 2024'
    >>> extract_signature_date(test_duplicates)
    '2024-11-01'
    >>> test_date_early = 'Date: November 10, 2023, SIGNATURES Pursuant to the requirements of Section 13 or 15(d) of the Securities Exchange Act of 1934, the Registrant has duly caused this report to be signed on its behalf by the undersigned, thereunto duly authorized. Date: November 1, 2023'
    >>> extract_signature_date(test_date_early)
    '2023-11-01'
    >>> test_no_signature = ' Date: November 1, 2024'
    >>> extract_signature_date(test_no_signature)
    '2024-11-01'
    >>> test_no_date = 'SIGNATURES'
    >>> extract_signature_date(test_no_date)
    'N/A'
    >>> test_no_date = 'as of November 2, 2024, SIGNATURES Date: November 1, 2024'
    >>> extract_signature_date(test_no_date)
    'N/A'
    """
    signature_pattern = r"SIGNATURES.*?Pursuant to the requirements of Section.*?(?=EXHIBIT INDEX|$)"
    signature_section_match = re.search(signature_pattern, text, re.DOTALL | re.IGNORECASE)

    if signature_section_match:
        signature_section = signature_section_match.group(0)
        signature_date_match = re.search(r"(?:Date:|as of)\s*(\w+\s\d{1,2}\s*,\s*\d{4})", signature_section)
        return convert_date(signature_date_match.group(1)) if signature_date_match else 'N/A'
    return 'N/A'


def extract_signers(text):
    """ 4. Who signed the report?
    If there are multiple signatures, all of them have to be listed (comma separated).
    Do not include the audit firm – in case it is given in the report.
    >>> duplicate_names = '/s/ Elon Musk Elon Musk Chief'
    >>> extract_signers(duplicate_names)
    ['Elon Musk']
    >>> duplicate_names = '/s/ Elon Musk Elon Musk Chief Executive Officer /s/ Elon Musk Chief Executive'
    >>> extract_signers(duplicate_names)
    ['Elon Musk', 'Elon Musk Chief']
    """
    cleaned_signers = []

    signer_pattern = r"/s/\s*([A-Z][a-zA-Z.\-]+\s[A-Z][a-zA-Z.\-]+(?:\s[A-Z][a-zA-Z.\-]+)?)"

    while True:
        match = re.search(signer_pattern, text)
        print(match)
        if not match:
            break
        signer = match.group(1).strip()

        # 检查是否有重复名字，例如 "Cesar Conde Cesar" -> "Cesar Conde"
        words = signer.split()
        if len(words) > 2 and words[-1].lower() == words[0].lower():  # 如果最后一个单词与第一个单词相同
            signer = ' '.join(words[:-1])

            # 避免重复，保持顺序
        if signer not in cleaned_signers:
            cleaned_signers.append(signer.strip())
        text = re.sub(re.escape(match.group(0)), '', text, flags=re.IGNORECASE)

    return cleaned_signers


LONG_MSFT = """
    <p style="font-size:10pt;margin-top:0;font-family:Times New Roman;margin-bottom:0;text-align:center;" id="signatures"><span style="color:#000000;white-space:pre-wrap;font-weight:bold;font-size:10pt;font-family:Arial;min-width:fit-content;">SIGNAT</span><span style="color:#000000;white-space:pre-wrap;font-weight:bold;font-size:10pt;font-family:Arial;min-width:fit-content;">URES</span></p>
  <p style="font-size:10pt;margin-top:9pt;font-family:Times New Roman;margin-bottom:0;text-align:justify;"><span style="color:#000000;white-space:pre-wrap;font-size:10pt;font-family:Arial;min-width:fit-content;">Pursuant to the requirements of Section 13 or 15(d) of the Securities Exchange Act of 1934, the Registrant has duly caused this report to be signed on its behalf by the undersigned; thereunto duly authorized, in the City of Redmond, State of Washington, on July 30, 2024.</span><span style="color:#000000;white-space:pre-wrap;font-size:10pt;font-family:Arial;min-width:fit-content;"> </span></p>
  <p style="font-size:10pt;margin-top:0;font-family:Times New Roman;margin-bottom:0;text-align:left;"><span style="white-space:pre-wrap;font-size:9pt;font-family:Arial;min-width:fit-content;">&#160;</span></p>
  <table style="border-spacing:0;table-layout:fixed;width:50.0%;border-collapse:separate;">
   <tr style="visibility:collapse;">
    <td style="width:100%;"/>
   </tr>
   <tr style="height:10pt;white-space:pre-wrap;word-break:break-word;text-align:right;">
    <td style="padding-top:0.01in;vertical-align:top;padding-right:0.01in;"><p style="font-size:10pt;margin-top:0;font-family:Times New Roman;margin-bottom:0;text-align:justify;"><span style="color:#000000;white-space:pre-wrap;font-family:Arial;min-width:fit-content;">M</span><span style="color:#000000;white-space:pre-wrap;font-size:7.5pt;font-family:Arial;min-width:fit-content;">ICROSOFT</span><span style="color:#000000;white-space:pre-wrap;font-family:Arial;min-width:fit-content;">&#160;C</span><span style="color:#000000;white-space:pre-wrap;font-size:7.5pt;font-family:Arial;min-width:fit-content;">ORPORATION</span></p></td>
   </tr>
   <tr style="white-space:pre-wrap;word-break:break-word;">
    <td style="padding-top:0.01in;vertical-align:middle;padding-right:0.01in;"><p style="font-size:9pt;margin-top:0;font-family:Times New Roman;margin-bottom:0;text-align:left;"><span style="white-space:pre-wrap;font-family:Arial;min-width:fit-content;">&#160;</span></p></td>
   </tr>
   <tr style="height:10pt;white-space:pre-wrap;word-break:break-word;text-align:right;">
    <td style="padding-top:0.01in;vertical-align:top;border-bottom:0.5pt solid;padding-right:0.01in;"><p style="font-size:10pt;margin-top:0;font-family:Times New Roman;margin-bottom:0;text-align:justify;"><span style="color:#000000;white-space:pre-wrap;font-family:Arial;min-width:fit-content;">/s/ A</span><span style="color:#000000;white-space:pre-wrap;font-size:7.5pt;font-family:Arial;min-width:fit-content;">LICE</span><span style="color:#000000;white-space:pre-wrap;font-family:Arial;min-width:fit-content;">&#160;L. J</span><span style="color:#000000;white-space:pre-wrap;font-size:7.5pt;font-family:Arial;min-width:fit-content;">OLLA</span></p></td>
   </tr>
    """

TSLA = """
SIGNATURES</span></div><div style="margin-top:12pt;text-indent:27.74pt"><span style="color:#000000;font-family:'Times New Roman',sans-serif;font-size:10pt;font-weight:400;line-height:120%">Pursuant to the requirements of Section 13 or 15(d) the Securities Exchange Act of 1934, the registrant has duly caused this report to be signed on its behalf by the undersigned, thereunto duly authorized.</span></div><div style="margin-top:12pt"><table style="border-collapse:collapse;display:inline-table;margin-bottom:5pt;vertical-align:text-bottom;width:100.000%"><tr><td style="width:1.0%"/><td style="width:58.445%"/><td style="width:0.1%"/><td style="width:1.0%"/><td style="width:39.355%"/><td style="width:0.1%"/></tr><tr><td colspan="3" style="padding:0 1pt"/><td colspan="3" style="padding:2px 1pt;text-align:center;vertical-align:bottom"><span style="color:#000000;font-family:'Times New Roman',sans-serif;font-size:10pt;font-weight:400;line-height:114%">Tesla, Inc.</span></td></tr><tr><td colspan="3" style="padding:0 1pt"/><td colspan="3" style="padding:2px 1pt;text-align:center;vertical-align:bottom"><span style="color:#000000;font-family:'Times New Roman',sans-serif;font-size:9pt;font-weight:400;line-height:114%">&#160;</span></td></tr><tr><td colspan="3" style="padding:2px 1pt;text-align:left;vertical-align:bottom"><div><span style="color:#000000;font-family:'Times New Roman',sans-serif;font-size:10pt;font-weight:400;line-height:114%">Date: January&#160;29, 2025</span></div></td><td colspan="3" style="padding:2px 1pt;text-align:center;vertical-align:bottom"><span style="color:#000000;font-family:'Times New Roman',sans-serif;font-size:10pt;font-weight:400;line-height:114%">/s/ Elon Musk</span></td></tr><tr><td colspan="3" style="padding:0 1pt"/><td colspan="3" style="border-top:0.5pt solid #000000;padding:2px 1pt;text-align:center;vertical-align:bottom"><span style="color:#000000;font-family:'Times New Roman',sans-serif;font-size:10pt;font-weight:400;line-height:114%">Elon Musk</span></td></tr><tr><td colspan="3" style="padding:0 1pt"/><td colspan="3" style="padding:2px 1pt;text-align:center;vertical-align:bottom"><span style="color:#000000;font-family:'Times New Roman',sans-serif;font-size:10pt;font-weight:400;line-height:114%">Chief Executive Officer</span></td></tr><tr><td colspan="3" style="padding:0 1pt"/><td colspan="3" style="padding:2px 1pt;text-align:center;vertical-align:bottom"><span style="color:#000000;font-family:'Times New Roman',sans-serif;font-size:10pt;font-weight:400;line-height:114%">(Principal Executive Officer)</span></td></tr></table></div><div style="margin-top:12pt;text-indent:27.74pt"><span style="color:#000000;font-family:'Times New Roman',sans-serif;font-size:10pt;font-weight:400;line-height:120%">Pursuant to the requirements of the Securities Exchange Act of 1934, this report has been signed below by the following persons on behalf of the registrant and in the capacities and on the dates indicated.</span></div><div style="margin-top:12pt"><table style="border-collapse:collapse;display:inline-table;margin-bottom:5pt;vertical-align:text-bottom;width:100.000%"><tr><td style="width:1.0%"/><td style="width:26.021%"/><td style="width:0.1%"/><td style="width:0.1%"/><td style="width:0.406%"/><td style="width:0.1%"/><td style="width:1.0%"/><td style="width:43.445%"/><td style="width:0.1%"/><td style="width:0.1%"/><td style="width:0.406%"/><td style="width:0.1%"/><td style="width:1.0%"/><td style="width:26.022%"/><td style="width:0.1%"/></tr><tr><td colspan="3" style="padding:2px 1pt;text-align:center;vertical-align:bottom"><span style="color:#000000;font-family:'Times New Roman',sans-serif;font-size:8pt;font-weight:700;line-height:114%">Signature</span></td><td colspan="3" style="padding:0 1pt"/><td colspan="3" style="padding:2px 1pt;text-align:center;vertical-align:bottom"><span style="color:#000000;font-family:'Times New Roman',sans-serif;font-size:8pt;font-weight:700;line-height:114%">Title</span></td><td colspan="3" style="padding:0 1pt"/><td colspan="3" style="padding:2px 1pt;text-align:center;vertical-align:bottom>
"""
