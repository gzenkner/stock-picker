import sys
sys.path.append('C:\\Users\\gabri\\my_projects\\stock_analysis')
from stock import TickerScraper

save_to = r'C:\Users\gabri\my_projects\stock_analysis\data'
wikiscraper = TickerScraper('sp500')
wikiscraper.get_sp500_csv(save_to, return_df=False, path_to_attributes=r'C:\Users\gabri\my_projects\stock_analysis\ticker_attributes.json')

