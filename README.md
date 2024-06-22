This library is collection of python classes (information and functions) designed to help individuals study stocks and improve stock picking and trading literacy. It pulls data from various sources, including Wikipedia and yFinance. It centers around the ticker (stock or share symbol) as the object and a ticker_attributes.json file, where you specify the information to extract such as a stock price timeseries or a stock summary. 

A scraper is used to retrieve key attributes (information), for example, a method (function) can be called from within the wikiScraper class that retrieves all Standard & Poor 500 companies, and writes high-level information to the ticker_attributes.json file. This includes the ticker and other information such as the founding date and sector it operates within. This is meant to simplify the process of collecting information on stocks of interest and could be extended to any number of indexes (collection of stocks), both within the US and globally.

    classes
        wikiScaper -> scrapes index information such as the S&P 500 and adds it to attributes
        yfScraper -> retrives specific information a ticker price or ticker summary
        StockAnalyzer -> a series of functions to analyse the timeseries data

To begin, identify stocks of interest, either by adding them directly to ticker_attributes.json file in the specified format, or running S&P500 scraper. Then, initialise a yfScraper class, which will load the ticker information to its attributes. You can then run various methods, such as get ticker summary information by passing in a list of tickers, or historical stock prices, again by passing in a list of tickers.

Once the data is gathered, you can start your analysis with the pre-built analytical tools in /analysis to analyse the market (S&P500), US senate trading records or individual stocks.


![S&P500 Sectors Heatmap](stock_analysis\images\Screenshot 2024-06-22 111302.png)
![S&P500 P/E Ratio Sectors](stock_analysis\images\Screenshot 2024-06-22 111349.png)
![S&P500 P/E Ratio, Beta and Market Cap in IT / Telecoms](stock_analysis\images\Screenshot 2024-06-22 111417.png)

