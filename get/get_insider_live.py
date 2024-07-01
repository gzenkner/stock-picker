from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

import pandas as pd
from datetime import datetime
import os


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

def make_df(data, path, start_page, end_page):
    scrape_date = datetime.now().date()
    df = pd.json_normalize(data)

    df['Stock'] = df['Stock'].apply(clean_text)
    df['Politician'] = df['Politician'].apply(clean_text)
    df['Transaction'] = df['Transaction'].apply(clean_text)
    df['Excess return *'] = df['Excess return *'].apply(clean_text)
    df['Traded'] = df['Traded'].apply(format_date)
    df['Filed'] = df['Filed'].apply(format_date)

    df['Transaction'] = df['Transaction'].str.replace('[\$,]', '', regex=True)
    df['Excess return *'] = df['Excess return *'].apply(lambda x: x.replace('%', '').strip() if isinstance(x, str) else x)

    df = df.rename(columns={'Stock': 'stock', 'Politician': 'politician', 'Transaction': 'transaction', 'Excess return *': 'excess_return', 'Traded': 'traded', 'Filed': 'filed'})

    file_name = f'insider_trades_{scrape_date}_pages_{start_page}-{end_page}.csv'
    file_path = os.path.join(path, file_name)
    df.to_csv(file_path, index=False)


start_page = 1
end_page = get_max_page_number(driver)
# end_page = 800
path = r'C:\Users\gabri\my_projects\stock_analysis\data'

all_data = []
for page in range(start_page, end_page):
    try:
        navigate_to_page(driver, page)
        data = scrape_page(driver)
        all_data.extend(data)
        print(f"Data from page {page}:")
        for entry in data:
            print(entry)
    except Exception as e:
        print(f"An error occurred on page {page}: {e}")

driver.quit()


make_df(all_data, path, start_page, end_page)