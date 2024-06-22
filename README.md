This library is collection of python classes (information and functions) designed to help individuals study stocks and improve stock picking and trading literacy. It pulls data from various sources, including Wikipedia and yFinance. It centers around the ticker (stock or share symbol) as the object and a ticker_attributes.json file, where you specify the information to extract such as a stock price timeseries or a stock summary. 

A scraper is used to retrieve key attributes (information), for example, a method (function) can be called from within the wikiScraper class that retrieves all Standard & Poor 500 companies, and writes high-level information to the ticker_attributes.json file. This includes the ticker and other information such as the founding date and sector it operates within. This is meant to simplify the process of collecting information on stocks of interest and could be extended to any number of indexes (collection of stocks), both within the US and globally.

    classes
        wikiScaper -> scrapes index information such as the S&P 500 and adds it to attributes
        yfScraper -> retrives specific information a ticker price or ticker summary
        StockAnalyzer -> a series of functions to analyse the timeseries data

To begin, initialise a yfScraper class