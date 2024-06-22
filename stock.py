import json
from datetime import datetime, UTC, timezone, timedelta
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import urllib.parse
import plotly.express as px
import os
import glob
import requests
from bs4 import BeautifulSoup
from typing import Optional

from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common import TimeoutException


class yfScraper:
    """
    A class that scrapes data from yf

    Attributes
    ----------
    tickers : json file
        A list of dictionaries, each containing information about a stock ticker.

    Methods
    -------
    __init__(tickers)
        Initializes the Scraper with the given tickers and base URL.
    """

    def __init__(self, ticker_attributes_path: Optional[str] = None):
        """
        Initializes the Scraper with the ticker attributes JSON file.

        Parameters:
        ticker_attributes_path (str, optional): Path to the ticker attributes JSON file. Defaults to False.
        
        Attributes:
        ticker_attributes_file (str): The absolute path to the ticker attributes JSON file.
        tickers (list): A list of tickers loaded from the JSON file.

        If the JSON file is not found, a new file with default content will be created.
        """
        default_content = [
            {
                "ticker": "NVDA", 
                "ticker_info": {
                    "security": None,
                    "gics_sector": None,
                    "gics_sub_industry": None,
                    "headquarters_location": None,
                    "date_added": None,
                    "cik": None,
                    "founded": None,
                },
                "ticker_summary": {
                    "date": None, 
                    "regular_market_price": None,
                    "regular_market_change": None,
                    "regular_market_change_percent": None,
                    "post_market_price": None,
                    "post_market_change": None,
                    "post_market_change_percent": None,
                    "previous_close": None,
                    "open_value": None,
                    "bid": None,
                    "ask": None,
                    "days_range": None,
                    "52_week_range": None,
                    "volume": None,
                    "avg_volume": None,
                    "market_cap": None,
                    "beta": None,
                    "pe_ratio": None,
                    "eps": None,
                    "earnings_date": None,
                    "forward_dividend_yield": None,
                    "ex_dividend_date": None,
                    "year_target_est": None,
                },
                "ticker_stock_price": {
                    "from_unix": None,
                    "to_unix": None,
                    "interval": None,
                    "frequency": None,
                    "from_date": None,
                    "to_date": None,
                    "url": None,
                    "raw_csv_destination": None,
                    "processed_csv_destination": None,
                    }
            }
            ]      
        if ticker_attributes_path:
            self.ticker_attributes_file = ticker_attributes_path
        else:
            matched_files = glob.glob(r'*ticker_attributes.json')
            self.ticker_attributes_file = os.path.abspath(matched_files[0]) if matched_files else os.path.abspath('ticker_attributes.json')
  
        if os.path.exists(self.ticker_attributes_file):
            print("File exists, successfully loaded attributes file: {}".format(self.ticker_attributes_file))
            with open(self.ticker_attributes_file, 'r') as f:
                data = json.load(f)  
    
            self.tickers = [i for i in data]
        else:
            print(f"Initialising 'ticker_attributes.json' as it was not found in this directory {__file__}")
            if not matched_files or os.path.getsize(self.ticker_attributes_file) == 0:
                with open(self.ticker_attributes_file, 'w') as file:
                    json.dump(default_content, file, indent=4)

            with open(self.ticker_attributes_file, 'r+') as file:
                attributes = json.load(file)
                file.seek(0)
                json.dump(attributes, file, indent=4)
                file.truncate()

    def update_json_file(self):
        self.ticker_attributes_file

        with open(self.ticker_attributes_file, 'r') as file:
            data = json.load(file)
        
        updated_data = []
        for item in data:
            for ticker_info in self.tickers:
                if item['ticker'] == ticker_info['ticker']:
                    item.update(ticker_info)
                    updated_data.append(item)
                    break
        
        with open(self.ticker_attributes_file, 'w') as file:
            json.dump(updated_data, file, indent=4)


    def attributes_df(self, tickers=None):
        """
        Converts the object's attributes derived from `ticker_attributes.json` file into a dataframe:
            - ticker_info: General information.
            - ticker_summary: Performance summary.
            - ticker_timeseries: Timeseries data.

        Returns:
            pd.DataFrame: A DataFrame with normalized ticker attributes.
        """
        if tickers:
            df = pd.json_normalize(self.tickers)
            return df[df['ticker'].isin(tickers)]
        else:
            return pd.json_normalize(self.tickers)
    

    def get_tickers(
            self, 
            source: str,
            return_df=False) -> None:
        """
        Retrieve stock tickers from the specified source by initialsing a TickerScraper instance and writing the tickers to the attributes files

        Parameters:
        -----------
        source : str
            The source of tickers: "sp500" or "senate".
        
        return_df : bool, optional
            If True, returns tickers as a DataFrame (default is False).

        Returns:
        --------
        None

        Example:
        --------
        get_tickers(source='sp500', return_df=True)
        """
        save_to = r'C:\\Users\\gabri\\my_projects\\stock_analysis\\data'
        if source == 'sp500':
            tickerscraper = TickerScraper(source)
            tickerscraper.get_sp500_csv(save_to, return_df, self.ticker_attributes_file)
        elif source == 'senate':
            pass

    def get_stock_summary(
        self, 
        tickers: list, 
        base_url: str = 'https://finance.yahoo.com/quote/'
        ) -> list:
        """
        Retrieves today's stock summary for a list of tickers from Yahoo Finance and updates the ticker_attributes.json file with the latest data.

        Parameters:
        tickers (list): A list of stock ticker symbols to retrieve summaries for.
        base_url (str): The base URL for Yahoo Finance stock quotes. Default is 'https://finance.yahoo.com/quote/'.

        Returns:
        list: A list of dictionaries, each containing the stock summary for a ticker.
        """
        ticker_list = []
        for ticker in tickers:

            options = Options()
            options.add_argument('--headless=new')
            driver = webdriver.Chrome(
                service=ChromeService(ChromeDriverManager().install()),
                options=options
            )
            driver.set_window_size(1150, 1000)
            url = base_url + urllib.parse.quote(ticker)
            print('Ticker: ', url)
            driver.get(url)
            try:
                consent_overlay = WebDriverWait(driver, 3).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, '.consent-overlay')))
                accept_all_button = consent_overlay.find_element(By.CSS_SELECTOR, '.accept-all')
                accept_all_button.click()
            except TimeoutException:
                print('Cookie consent overlay missing')

            def safe_extract(by, selector):
                try:
                    return driver.find_element(by, selector).text
                except Exception:
                    return 'not available'
                
            ticker_summary = {'ticker': ticker, 'summary': {}}
            ticker_summary['summary']['date'] = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")

            ticker_summary['summary']['regular_market_price'] = safe_extract(By.CSS_SELECTOR, f'[data-symbol="{ticker}"][data-field="regularMarketPrice"]')
            ticker_summary['summary']['regular_market_change'] = safe_extract(By.CSS_SELECTOR, f'[data-symbol="{ticker}"][data-field="regularMarketChange"]')
            ticker_summary['summary']['regular_market_change_percent'] = safe_extract(By.CSS_SELECTOR, f'[data-symbol="{ticker}"][data-field="regularMarketChangePercent"]')
            ticker_summary['summary']['post_market_price'] = safe_extract(By.CSS_SELECTOR, f'[data-symbol="{ticker}"][data-field="postMarketPrice"]')
            ticker_summary['summary']['post_market_change'] = safe_extract(By.CSS_SELECTOR, f'[data-symbol="{ticker}"][data-field="postMarketChange"]')
            ticker_summary['summary']['post_market_change_percent'] = safe_extract(By.CSS_SELECTOR, f'[data-symbol="{ticker}"][data-field="postMarketChangePercent"]')
            ticker_summary['summary']['previous_close'] = safe_extract(By.CSS_SELECTOR, 'fin-streamer[active][data-field="regularMarketPreviousClose"].svelte-tx3nkj')
            ticker_summary['summary']['open_value'] = safe_extract(By.CSS_SELECTOR, 'fin-streamer[active][data-field="regularMarketOpen"].svelte-tx3nkj')
            ticker_summary['summary']['bid'] = safe_extract(By.XPATH, '//span[contains(text(), "Bid")]/following-sibling::span[@class="value svelte-tx3nkj"]')
            ticker_summary['summary']['ask'] = safe_extract(By.XPATH, '//span[contains(text(), "Ask")]/following-sibling::span[@class="value svelte-tx3nkj"]')
            ticker_summary['summary']['days_range'] = safe_extract(By.CSS_SELECTOR, 'fin-streamer[active][data-field="regularMarketDayRange"].svelte-tx3nkj')
            ticker_summary['summary']['52_week_range'] = safe_extract(By.CSS_SELECTOR, 'fin-streamer[active][data-field="fiftyTwoWeekRange"].svelte-tx3nkj')
            ticker_summary['summary']['volume'] = safe_extract(By.CSS_SELECTOR, 'fin-streamer[active][data-field="regularMarketVolume"].svelte-tx3nkj')
            ticker_summary['summary']['avg_volume'] = safe_extract(By.CSS_SELECTOR, 'fin-streamer[active][data-field="averageVolume"].svelte-tx3nkj')
            ticker_summary['summary']['market_cap'] = safe_extract(By.CSS_SELECTOR, 'fin-streamer[active][data-field="marketCap"].svelte-tx3nkj')
            ticker_summary['summary']['beta'] = safe_extract(By.XPATH, '//span[contains(text(), "Beta (5Y Monthly")]/following-sibling::span[@class="value svelte-tx3nkj"]')
            ticker_summary['summary']['pe_ratio'] = safe_extract(By.XPATH, '//span[contains(text(), "PE Ratio (TTM)")]//following-sibling::span[@class="value svelte-tx3nkj"]')
            ticker_summary['summary']['eps'] = safe_extract(By.XPATH, '//span[contains(text(), "EPS (TTM)")]//following-sibling::span[@class="value svelte-tx3nkj"]/fin-streamer')
            ticker_summary['summary']['earnings_date'] = safe_extract(By.XPATH, '//span[contains(text(), "Earnings Date")]/following-sibling::span[@class="value svelte-tx3nkj"]')
            ticker_summary['summary']['forward_dividend_yield'] = safe_extract(By.XPATH, '//span[contains(text(), "Forward Dividend & Yield")]/following-sibling::span[@class="value svelte-tx3nkj"]')
            ticker_summary['summary']['ex_dividend_date'] = safe_extract(By.XPATH, '//span[contains(text(), "Ex-Dividend Date")]/following-sibling::span[@class="value svelte-tx3nkj"]')
            ticker_summary['summary']['year_target_est'] = safe_extract(By.XPATH, '//span[contains(text(), "1y Target Est")]/following-sibling::span[@class="value svelte-tx3nkj"]')


            for i in self.tickers:
                if i['ticker'] == ticker:
                        i['ticker_summary'] = ticker_summary['summary']

            ticker_list.append(ticker_summary)

        return ticker_list


    def get_stock_price(
        self, 
        tickers: list, 
        destination: str, 
        from_unix: int,
        to_unix: int,
        base_url: str = 'https://finance.yahoo.com/quote/',
        frequency = '1d',
        interval = '1d',
        ) -> None:
        """
        Scrapes historical data for a ticker and saves it locally as CSV
        """

        stock_prices = []
        for ticker in tickers:
            stock_price = {'ticker':ticker, 'ticker_stock_price':{}}
            epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
            stock_price['ticker_stock_price']['to_unix'] = to_unix
            stock_price['ticker_stock_price']['from_unix'] = from_unix
            stock_price['ticker_stock_price']['from_date'] = (epoch + timedelta(seconds=stock_price['ticker_stock_price']['from_unix'])).strftime('%Y-%m-%d')
            stock_price['ticker_stock_price']['to_date'] = (epoch + timedelta(seconds=stock_price['ticker_stock_price']['to_unix'])).strftime('%Y-%m-%d')
            stock_price['ticker_stock_price']['interval'] = interval
            stock_price['ticker_stock_price']['frequency'] = frequency
            url = (
                f"{base_url}/{urllib.parse.quote(ticker)}/history?"
                f"period1={from_unix}&"
                f"period2={to_unix}&"
                f"interval={interval}&"
                f"filter=history&"
                f"frequency={frequency}&"
                f"includeAdjustedClose=true"
            )
            print(url)
            stock_price['ticker_stock_price']['url'] = url
            # stock_price['ticker_stock_price']['raw_csv_destination'] = destination
            stock_prices.append(stock_price)
            
            p = Path(destination) / Path(f"{ticker}_{stock_price['ticker_stock_price']['frequency']}_{stock_price['ticker_stock_price']['from_date']}_{stock_price['ticker_stock_price']['to_date']}.csv")
            stock_price['ticker_stock_price']['raw_csv_destination'] = str(p)


            # write to attributes
            for i in self.tickers:
                if i['ticker'] == ticker:
                        i['ticker_stock_price'] = stock_price['ticker_stock_price'] 

            # write to attributes JSON file 
            self.update_json_file()


            options = Options()
            options.add_argument('--headless=new')
            driver = webdriver.Chrome(
                service=ChromeService(ChromeDriverManager().install()),
                options=options
            )
            driver.set_window_size(1150, 1000)
            driver.get(url)

            try:
                consent_overlay = WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, '.consent-overlay')))
                accept_all_button = consent_overlay.find_element(By.CSS_SELECTOR, '.accept-all')
                accept_all_button.click()
            except TimeoutException:
                print('Cookie consent overlay missing')

            try:
                table_element = driver.find_element(By.CLASS_NAME, 'table-container.svelte-ewueuo')
                rows = table_element.find_elements(By.TAG_NAME, 'tr')

                if len(rows) == 0:
                    print('Dataframe has no records, unable to process the file')

                headers = []
                table_data = []
                for index, row in enumerate(rows):
                    if index == 0:
                        headers = [cell.text for cell in row.find_elements(By.TAG_NAME, 'th')]
                    else:
                        cells = row.find_elements(By.TAG_NAME, 'td')
                        row_data = [cell.text for cell in cells]
                        table_data.append(row_data)

                driver.quit()

                df = pd.DataFrame(columns=headers, data=table_data)
                print('Saving to: ', p)
                df.to_csv(p, index=False)
            except Exception as e:
                print('Error:', e)

    def stock_price_post_processing(self, ticker, destination):
        attributes = [i for i in self.tickers if i['ticker'] == ticker][0]['ticker_stock_price']
        df = pd.read_csv(attributes['raw_csv_destination'])
        df['Date'] = pd.to_datetime(df['Date'], format='%b %d, %Y')
        df.columns = [col.lower().replace(' ', '_') for col in df.columns]
        df['ticker'] = ticker
                
        numeric_columns = ['open', 'high', 'low', 'close', 'adj_close', 'volume']
        for column in numeric_columns:
            if column in df.columns:
                df[column] = df[column].astype(str).str.replace(',', '')
        df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')
        
        p = Path(destination) / Path(f'{ticker}_{attributes['frequency']}_{attributes['from_date']}_{attributes['to_date']}_processed.csv')
        df.to_csv(p, index=False)

        for i in self.tickers:
            if i['ticker'] == ticker:
                    i['ticker_stock_price']['processed_csv_destination'] = str(p)
                    
        self.update_json_file()

        

    def get_stock_price_dfs(
            self, 
            tickers: list,
            ) -> pd.DataFrame:
        paths = [i['ticker_stock_price']['processed_csv_destination'] for i in self.tickers if i['ticker'] in tickers]
        dfs = [pd.read_csv(path) for path in paths]
        concatenated_df = pd.concat(dfs, ignore_index=True)
        return concatenated_df

class StockAnalyzer:
    def __init__(self, yfScraper):
        self.yfScraper = yfScraper

    def line_plot(
            self, 
            tickers: list
            ):
    
        fig = px.line(
            data_frame = self.yfScraper.get_stock_price_dfs(tickers),
            x='date', 
            y='close', 
            color='ticker'
            )
        return fig.show()


    def candlestick_plot(self, df):
        fig = go.Figure(
            data = [  
                go.Candlestick(
                    x=df['date'],
                    open=df['open'],
                    high=df['high'],
                    low=df['low'],
                    close=df['close']
                )
                ]
            )

        fig.show()

    def resample_timeseries(self, df, period='ME', column='close', ascending=False):
        """Resample time series data by a specified period.

        Args:
        df (pd.DataFrame): DataFrame containing the time series data.
        period (str): Resampling period (e.g., 'Y', 'M', 'W', 'D', 'YS', 'YE').
        column (str): The column to resample (e.g., 'close').

        Returns:
        pd.DataFrame: Resampled DataFrame with the specified period.
        """
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        
        df = df.sort_values('date', ascending=False)
        df.set_index('date', inplace=True)
        all_tickers = df['ticker'].unique()
        monthly_close_all = pd.DataFrame()

        for ticker in all_tickers:
            filtered_df = df[df['ticker'] == ticker].copy()
            monthly_close = filtered_df[column].resample(period).mean()
            monthly_close = monthly_close.rename(ticker)
            monthly_close.index = monthly_close.index.strftime('%Y-%m-%d')
            monthly_close_all = pd.concat([monthly_close_all, monthly_close], axis=1)
        
        monthly_close_all = monthly_close_all.sort_index(ascending=ascending)

        return monthly_close_all

    def percentage_change(self, df: pd.DataFrame) -> dict:
        """
        Calculate the percentage change for each column in a DataFrame and 
        return the average percentage change for each column as a dictionary.

        Parameters:
        -----------
        df : pandas.DataFrame
            The input DataFrame for which the percentage change needs to be calculated. 
            Each column in the DataFrame is processed individually.

        Returns:
        --------
        dict
            A dictionary where the keys are the column names of the DataFrame and the values 
            are the average percentage change for each column, calculated while skipping NaN values.

        Method:
        -------
        1. The method first calculates the percentage change for each column in the DataFrame.
        This is done using the `pct_change()` method of pandas, which computes the percentage 
        change between the current and prior element.
        2. The resulting percentage change values are multiplied by 100 to convert them to a percentage.
        3. An empty dictionary, `average_percentage_change`, is initialized to store the average 
        percentage change for each column.
        4. The method iterates over each column in the percentage change DataFrame.
        5. For each column, the mean of the percentage change values is calculated, 
        ignoring NaN values using the `mean(skipna=True)` method.
        6. The average percentage change for each column is stored in the `average_percentage_change` dictionary.
        7. Finally, the method returns the `average_percentage_change` dictionary.

        Example:
        --------
        >>> df = pd.DataFrame({
        >>>     'A': [100, 200, 300, 400],
        >>>     'B': [50, 60, 70, 80]
        >>> })
        >>> obj = YourClassName()
        >>> obj.percentage_change(df)
        {'A': 50.0, 'B': 20.0}

        Notes:
        ------
        - The `fill_method=None` argument in `pct_change()` is used to ensure that no forward or 
        backward filling is applied before calculating the percentage change.
        - This method assumes that `df` is a pandas DataFrame and that its columns contain numerical data.
        """
        percentage_change = df.pct_change(fill_method=None) * 100
        average_percentage_change = {}

        for column in percentage_change.columns:
            avg_rate = percentage_change[column].mean(skipna=True)
            average_percentage_change[column] = avg_rate

        return average_percentage_change


    def bar_chart_pct_change(self, df, column='^GSPC'):
        df_new = df[column].pct_change(fill_method=None).dropna() * 100
        fig = px.bar(x=df_new.index, y=df_new.values, title=column)
        fig.update_layout(width=800, height=300)
        fig.show()


    def cross_correlation_matrix(self, df, method='pearson'):
        "method : {'pearson', 'kendall', 'spearman'} or callable"
        df = df.sort_index()
        return df.corr(method)
    
    
class TickerScraper:
    """
    A class to scrape the list of S&P 500 companies from Wikipedia and perform various operations.

    Attributes
    ----------
    name : str
        The name of the TickerScraper instance.

    Methods
    -------
    get_sp500_csv(destination, return_df=True, path_to_attributes=None)
        Scrapes the S&P 500 companies table from Wikipedia and saves it as a CSV file. Optionally returns the DataFrame and updates a JSON file with ticker information.
    """

    def __init__(self, name):
        """
        Constructs all the necessary attributes for the TickerScraper object.

        Parameters
        ----------
        name : str
            The name of the TickerScraper instance.
        """
        self.name = name

    def get_sp500_csv(self, destination, return_df=True, path_to_attributes=None):
        """
        Scrapes the S&P 500 companies table from Wikipedia and saves it as a CSV file. Optionally returns the DataFrame and updates a JSON file with ticker information.

        Parameters
        ----------
        destination : str
            The directory where the CSV file will be saved.
        return_df : bool, optional
            If True, returns the DataFrame containing the S&P 500 companies (default is True).
        path_to_attributes : str, optional
            The path to a JSON file containing existing ticker information. If provided, the method updates this file with new ticker information (default is None).

        Returns
        -------
        pandas.DataFrame
            DataFrame containing the S&P 500 companies if return_df is True.
        """
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find('table', {'id': 'constituents'})

        headers = []
        for th in table.find_all('th'):
            headers.append(th.text.strip())

        rows = []
        for tr in table.find_all('tr')[1:]:
            cells = tr.find_all('td')
            row = [cell.text.strip() for cell in cells]
            rows.append(row)

        df = pd.DataFrame(rows, columns=headers)
        p = Path(destination) / f'sp500_companies_{datetime.now().date()}.csv'
        df.to_csv(p, index=False)
        if return_df:
            return df
        if path_to_attributes:
            print('Writing to object attributes: ', path_to_attributes)
            with open(path_to_attributes, 'r') as file:
                existing_tickers = json.load(file)

            existing_symbols = {item['ticker'] for item in existing_tickers}
            new_symbols = df['Symbol']
            ticker_info_df = df.copy()
            ticker_info_df.columns = ticker_info_df.columns.str.lower().str.replace(' ', '_').str.replace('.', '_').str.replace('-', '_')

            new_tickers = []
            for symbol in new_symbols:
                ticker_info_row = ticker_info_df[ticker_info_df['symbol'] == symbol].iloc[0].to_dict()
                del ticker_info_row['symbol'] 
                
                if symbol not in existing_symbols:
                    ticker_entry = {
                        "ticker": symbol,
                        "ticker_info": ticker_info_row  
                    }
                    new_tickers.append(ticker_entry)
                else:
                    for item in existing_tickers:
                        if item['ticker'] == symbol:
                            item['ticker_info'] = ticker_info_row
                            break

            existing_tickers.extend(new_tickers)
            with open(path_to_attributes, 'w') as file:
                json.dump(existing_tickers, file, indent=4)







    # def get_stock_ipo(is_valid_guess, start, end):
    #     """
    #     Binary search to find the Unix timestamp, with a count of the number of guesses made.
        
    #     :param is_valid_guess: Function to check if a guess is valid.
    #     :param start: The starting point of the search range (Unix timestamp).
    #     :param end: The ending point of the search range (Unix timestamp).
    #     :return: Tuple containing the valid Unix timestamp and the number of guesses made.
    #     """
    #     # Ensure the range is a multiple of 86400 seconds (one day)
    #     start -= start % 86400
    #     end -= end % 86400
        
    #     guess_count = 0
        
    #     while start <= end:
    #         guess_count += 1
    #         mid = start + ((end - start) // 86400 // 2) * 86400  # Midpoint in increments of one day
            
    #         if is_valid_guess(mid):
    #             end = mid - 86400  # Move to the lower half
    #         else:
    #             start = mid + 86400  # Move to the upper half
        
    #     return start, guess_count

    # # Example usage with a mock validation function:
    # def is_valid_guess(guess):
    #     target = 916963200  # nvda
    #     target = -1325635200 # gspc
    #     return guess >= target

    # # Unix timestamps for January 1, 1990 and January 1 of the following year
    # start_timestamp = int(datetime(1990, 1, 1).timestamp())
    # next_year = datetime.now().year + 2
    # end_timestamp = int(datetime(next_year, 1, 1).timestamp())

    # # Find the Unix timestamp and number of guesses
    # result, num_guesses = find_unix_timestamp_with_guess_count(is_valid_guess, start_timestamp, end_timestamp)
    # print("Valid Unix timestamp:", result)
    # print("Number of guesses made:", num_guesses)

    # # Convert the result to a human-readable date
    # result_date = datetime.fromtimestamp(result)
    # print("Valid date:", result_date.strftime('%Y-%m-%d'))


class UpdateScraper():
    """
    A class that updates the scraper when it notices errors getting CSS stuff or whatever
    This utilizes Llama3 to repair broken tags in the scraper making this class self-healing
    """
    def __init__(self, name):
        self.name = name
        pass

    def update_get_stock_price(self):
        params = {}
        pass

    def update_get_stock_summary(self):
        params = {'bid':'bid', 'ask':'ask'}
        pass



