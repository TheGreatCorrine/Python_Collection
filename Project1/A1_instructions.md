S1. Download all 10-k files from the SEC website: [sec_companysearch](https://www.sec.gov/edgar/searchedgar/companysearch)

| **Company Name**           | **Ticker** | **CIK**      |
|----------------------------|--------------|--------------|
| Apple Inc.                | AAPL         | 0000320193   | 2024
| Microsoft Corporation     | MSFT         | 0000789019   | 
| Amazon.com, Inc.          | AMZN         | 0001018724   |
| Tesla, Inc.               | TSLA         | 0001318605   |
| Meta Platforms, Inc.      | META         | 0001326801   | 2025
| Alphabet Inc.             | GOOGL        | 0001652044   |
| Johnson & Johnson         | JNJ          | 0000200406   |
| The Coca-Cola Company     | KO           | 0000021344   | 2024
| Procter & Gamble Co.      | PG           | 0000080424   |
| Walmart Inc.              | WMT          | 0000104169   |

So far, we've downloaded __8__ companies.

Save all files as plain text (.html) and rename them in this format: __TICKER_10K__
These HTML files should be the only HTML files we submit.

The `10-k html files` folder contains all the 10-k files from the 10 companies.
Please upload them here.

----------------
## Some Python libraries you use:
1. Pathlib
```bash
  from pathlib import Path
```
Another way to iterate all files in a folder:
```bash
    for file in Path('10-k html files').iterdir(): 
    # iterates through all files in the folder.
```
2. 
