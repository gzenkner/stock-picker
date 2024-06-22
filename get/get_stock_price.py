import concurrent.futures
import threading
import time
import json

import sys
sys.path.append('C:\\Users\\gabri\\my_projects\\stock_analysis')
from stock import yfScraper
from utils import print_elapsed_time

def process_ticker(ticker, destination, from_unix, to_unix):
    if ticker != "^GSPC":
        try:
            scrape.get_stock_price([ticker], destination, from_unix, to_unix)
            scrape.stock_price_post_processing(ticker, destination)
            print('g')
        except Exception as e:
            print(f'Error processing {ticker}: {e}')

start_time = time.time()
threading.Thread(target=print_elapsed_time, args=(15,), daemon=True).start()

scrape = yfScraper(r'C:\Users\gabri\my_projects\stock_analysis\ticker_attributes.json')

tickers = ['AVGO']
from_unix = 1249516800
to_unix = 1719014400
csv_destination = r'C:\Users\gabri\my_projects\stock_analysis\data'


with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
    executor.map(lambda ticker: process_ticker(ticker, csv_destination, from_unix, to_unix), tickers)

elapsed_time = time.time() - start_time
print(f"Final elapsed time: {int(elapsed_time)} seconds")
