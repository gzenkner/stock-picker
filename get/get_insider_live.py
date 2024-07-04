from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

import pandas as pd
from datetime import datetime
import os
import re


options = webdriver.ChromeOptions()
options.add_argument('--headless')  # Enable headless mode
options.add_argument('--disable-gpu')  # Disable GPU acceleration (necessary for headless mode)
options.add_argument('--no-sandbox')  # Bypass OS security model (useful for certain environments)
options.add_argument('--disable-dev-shm-usage')  # Overcome limited resource problems in Docker

service = ChromeService(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=options)

url = 'https://trendspider.com/markets/congress-trading/'
driver.get(url)


def scrape_page(driver):
    table_body = WebDriverWait(driver, 2).until(  
        EC.presence_of_element_located((By.XPATH, '//tbody[@class="data-table__body" and @x-ref="table"]'))
    )
    data = []
    rows = table_body.find_elements(By.CLASS_NAME, 'data-table__row')
    for row in rows:
        row_data = {}
        cells = row.find_elements(By.CLASS_NAME, 'data-table__cell')
        for cell in cells:
            data_title = cell.get_attribute('data-title')
            cell_text = cell.text
            row_data[data_title] = cell_text
        data.append(row_data)
    return data


def navigate_to_page(driver, page_number):
    try:
        current_page_xpath = f'//a[@aria-label="Current Page, Page {page_number}."]'
        if driver.find_elements(By.XPATH, current_page_xpath):
            print(f"Already on page {page_number}")
            return

        pagination_buttons = driver.find_elements(By.XPATH, '//a[contains(@aria-label, "Go to page")]')
        for button in pagination_buttons:
            print(f"Found button: {button.get_attribute('aria-label')}")

        page_button_xpath = f'//a[@aria-label="Go to page {page_number}."]'
        next_page_button = WebDriverWait(driver, 2).until(  
            EC.element_to_be_clickable((By.XPATH, page_button_xpath))
        )
        driver.execute_script("arguments[0].click();", next_page_button)
        WebDriverWait(driver, 2).until(  
            EC.presence_of_element_located((By.XPATH, f'//a[@aria-label="Current Page, Page {page_number}."]'))
        )
        print(f"Successfully navigated to page {page_number}")
    except Exception as e:
        print(f"Failed to navigate to page {page_number}: {e}")
        raise


def get_max_page_number(driver):
    try:
        # Find all pagination buttons
        pagination_buttons = driver.find_elements(By.XPATH, '//a[contains(@aria-label, "Go to page")]')
        max_page_number = 1
        for button in pagination_buttons:
            label = button.get_attribute('aria-label')
            if 'Go to page' in label:
                page_number = int(label.split()[-1].strip('.'))
                if page_number > max_page_number:
                    max_page_number = page_number
        return max_page_number
    except Exception as e:
        print(f"Failed to determine the maximum page number: {e}")
        raise

def clean_text(text):
    return text.split('\n')[0].strip() if isinstance(text, str) else text

def format_date(date_str):
    return datetime.strptime(date_str.strip(), '%b %d, %Y').strftime('%Y-%m-%d') if isinstance(date_str, str) else date_str

def post_process_insider(raw_file_path, processed_file_path):
    df = pd.read_csv(raw_file_path)
    
    tickers = []
    companies = []
    representatives = []
    parties = []
    transaction_types = []
    transaction_amounts_min = []
    transaction_amounts_max = []
    excess_returns = []
    traded_dates = []
    filed_dates = []
    
    def parse_transaction_amount(transaction_amount):
        amounts = re.sub(r'[^\d\-]', '', transaction_amount).split('-')
        amount_min = int(amounts[0])
        amount_max = int(amounts[1]) if len(amounts) > 1 else amount_min
        return amount_min, amount_max

    def parse_excess_return(excess_return):
        if excess_return.strip() == '-' or not excess_return.strip():
            return None
        excess_return_cleaned = excess_return.replace(',', '')  # Remove commas
        return float(excess_return_cleaned.strip('%'))

    for index, row in df.iterrows():
        stock_details = row['Stock'].split('\n')
        ticker = stock_details[0]  # Extract the ticker
        company = stock_details[1] if len(stock_details) > 1 else ""  # Extract the company name
        
        politician_details = row['Politician'].split('\n')
        representative = politician_details[0]  # Extract the representative's name
        party = politician_details[1] if len(politician_details) > 1 else ""  # Extract the party
        
        transaction_details = row['Transaction'].split('\n')
        transaction_type = transaction_details[0]
        transaction_amount = transaction_details[1]
        
        amount_min, amount_max = parse_transaction_amount(transaction_amount)
    
        excess_return = parse_excess_return(row['Excess return *'])
        
        traded_date = pd.to_datetime(row['Traded']).strftime('%Y-%m-%d')
        filed_date = pd.to_datetime(row['Filed']).strftime('%Y-%m-%d')
        
        tickers.append(ticker)
        companies.append(company)
        representatives.append(representative)
        parties.append(party)
        transaction_types.append(transaction_type)
        transaction_amounts_min.append(amount_min)
        transaction_amounts_max.append(amount_max)
        excess_returns.append(excess_return)
        traded_dates.append(traded_date)
        filed_dates.append(filed_date)
    
    parsed_df = pd.DataFrame({
        'ticker': tickers,
        'company': companies,
        'representative': representatives,
        'party': parties,
        'transaction_type': transaction_types,
        'transaction_amount_min': transaction_amounts_min,
        'transaction_amount_max': transaction_amounts_max,
        'excess_return_perc': excess_returns,
        'traded': traded_dates,
        'filed': filed_dates
    })

    parsed_df['transaction_amount_min'] = pd.to_numeric(parsed_df['transaction_amount_min'])
    parsed_df['transaction_amount_max'] = pd.to_numeric(parsed_df['transaction_amount_max'])
    parsed_df['excess_return_perc'] = pd.to_numeric(parsed_df['excess_return_perc'])

    parsed_df['traded'] = pd.to_datetime(parsed_df['traded'])
    parsed_df['filed'] = pd.to_datetime(parsed_df['filed'])

    parsed_df.to_csv(processed_file_path, index=False)
    print(f'Saved processed file to: {processed_file_path}')


start_page = 1
end_page = get_max_page_number(driver)


all_data = []
for page in range(start_page, end_page):
    try:
        navigate_to_page(driver, page)
        data = scrape_page(driver)
        all_data.extend(data)

    except Exception as e:
        print(f"An error occurred on page {page}: {e}")


scrape_date = datetime.now().date()
data_dir = r'C:\Users\gabri\my_projects\stock_analysis\data'
raw_file_name = f'insider_trades_{scrape_date}_pages_{start_page}-{end_page}_raw.csv'
raw_file_path = os.path.join(data_dir, raw_file_name)

df = pd.json_normalize(all_data)
df.to_csv(raw_file_path, index=False)
print(f'Saved raw file to: {raw_file_path}')

driver.quit()


processed_file_name = f'insider_trades_{scrape_date}_pages_{start_page}-{end_page}.csv'
processed_file_path = os.path.join(data_dir, processed_file_name)
try:
    post_process_insider(raw_file_path, processed_file_path)
except Exception as e:
    print(f'Error, failed to process all entries: {e}')
