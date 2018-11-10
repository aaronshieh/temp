from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from bs4 import BeautifulSoup
import requests
import numpy as np
import pandas as pd
import datetime
from . import helpers

# Create your views here.
def getHistoricData(request):
    start_date = '20130428' # earliest: 20130428, earliest with volume: 20131227
    end_date = datetime.datetime.now().strftime("%Y%m%d") # '20181110'
    url = f'https://coinmarketcap.com/currencies/bitcoin/historical-data/?start={start_date}&end={end_date}'

    header = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/69.0.3497.100 Safari/537.36'
    }

    response = requests.get(url, headers=header)
    soup = BeautifulSoup(response.text, "lxml")

    historical_data = soup.select('#historical-data table')[0]
    tr = list(historical_data.find_all('tr'))
    column_labels = tr[0].get_text().strip().replace('*','').split('\n')

    # try to open csv file
    try:
        print("opening csv...")
        df = pd.read_csv('bitcoin_historical_data.csv')
        df['Date'] = pd.to_datetime(df['Date'])
        print("[success] finished loading csv.")
        
        # fetch latest record from scraped html
        latest = tr[1].get_text().strip().replace(',','').split('\n')
        df_temp = pd.DataFrame([latest], columns=column_labels)
        df_temp['Date'] = pd.to_datetime(df_temp['Date'])

        # compare dates & update data if necessary
        df_temp_date = df_temp.at[0,'Date']
        if df.iloc[-1]['Date'] == df_temp_date:
            print('already up-to-date!')
        else:
            print('new data available, updating csv...')
            helpers.format_data(df_temp)
            df = pd.concat([df,df_temp], ignore_index=True)
            
            df.to_csv('bitcoin_historical_data.csv', index=False)
        
    except FileNotFoundError:
        print("[FileNotFoundError] create csv file...")
        df = pd.DataFrame()
        for x in range(len(tr)):
            if x != 0:
                row_data = tr[x].get_text().strip().replace(',','').split('\n')
                df = pd.concat([df,pd.DataFrame([row_data], columns=column_labels)], ignore_index=True)

        df['Date'] = pd.to_datetime(df['Date'])
        helpers.format_data(df)
        
        df.sort_values('Date', ascending=True, inplace=True)
        df.reset_index(drop=True, inplace=True)
        print(df.head())
        print(df.tail())
        print(df.dtypes)
        print(df.describe())
        
        try:
            print('writing to file...')
            df.to_csv('bitcoin_historical_data.csv', index=False)
            print('[success] finished writing to file.')
        except PermissionError:
            print('[PermissionError] failed writing to file! ')

    return HttpResponse(historical_data.prettify())


    # article_list = []
    # articles = soup.select('.page-content .posts-row article')
    # for article in articles:
    #     article_title = article.find('header').find('h4', class_="entry-title").find('a').string
    #     article_link = article.find('header').find('h4', class_="entry-title").find('a').get('href')
    #     article_image = article.find('div', class_='post-thumbnail').find('img').get('src')
    #     article_date = article.find('header').find(class_="updated").get('datetime')
    #     article_list.append({"title":article_title, "link":article_link, "img":article_image, "time":article_date})
    #     # print(article_title)
    #     # print(article_link)
    #     # print(article_image)
    #     # print(article_date)
    #     # print('')

    # articles_ = {"articles":article_list}
    # # articles_ = json.dumps(articles_)
    # # print(articles_)
    # return JsonResponse(articles_)