import requests
import json
import datetime

save_to = f'C:\\Users\\gabri\\my_projects\\stock_analysis\\data\\senate_stock_tracker_{datetime.datetime.now().date()}.json'
url = 'https://house-stock-watcher-data.s3-us-west-2.amazonaws.com/data/all_transactions.json'
response = requests.get(url)
data = response.json()

with open(save_to, 'w') as file:
    json.dump(data, file) 