import sys

sys.path.append('C:\\Users\\gabri\\my_projects\\stock_analysis')
import time
import concurrent.futures
from stock import yfScraper


def get_tickers(tickers, most_up_to_date_first=False, specific_tickers=False, get_empty_tickers=False, gics_sub_industry=False):
    """
    Get tickers in an order that makes sense, 
    1) typically oldest to newest, 
    2) empty tickers when new tickers are added to attributes
    3) a specific set you are interested in
    """
    if specific_tickers:
        print(f'Getting: {specific_tickers}')
        if not isinstance(specific_tickers, list):
            return [specific_tickers]
        else: 
            return specific_tickers
    if get_empty_tickers:
        return [i['ticker'] for i in scrape.tickers if 'ticker_summary' not in i]
    if gics_sub_industry:
        tickers = [i['ticker'] for i in scrape.tickers if i['ticker'] !='^GSPC' and i['ticker_info']['gics_sub_industry'] == 'Aerospace & Defense']
        print(f'Getting: {tickers}')
        return tickers

    tickers_dates = [(i['ticker'], i['ticker_summary']['date']) for i in tickers if i['ticker'] != '^GSPC']
    sorted_tickers_dates = sorted(tickers_dates, key=lambda x: x[1], reverse=most_up_to_date_first)
    sorted_tickers = [ticker for ticker, date in sorted_tickers_dates]
    return sorted_tickers

def process_ticker(ticker):
    if ticker != "^GSPC":
        try:
            scrape.get_stock_summary([ticker])
            scrape.update_json_file()
        except Exception as e:
            print(f'Error processing {ticker}: {e}')

start_time = time.time()


scrape = yfScraper(r'C:\Users\gabri\my_projects\stock_analysis\ticker_attributes.json')
tickers = get_tickers(scrape.tickers, most_up_to_date_first=False, specific_tickers=False)
# tickers = tickers[0:5]


with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
    executor.map(process_ticker, tickers)

elapsed_time = time.time() - start_time
print(f"Elapsed time: {int(elapsed_time)} seconds")


