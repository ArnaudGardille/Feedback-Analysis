from bs4 import BeautifulSoup
import requests
import pandas as pd
from time import sleep
from tqdm import trange
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--company', type=str, nargs=1)
parser.add_argument('--language', type=str, nargs='?', default='fr')
parser.add_argument('--from-page', type=int, nargs='?', default=1)
parser.add_argument('--to-page', type=int, nargs='?', default=1)
args = parser.parse_args()


def soup2list(src, list_, attr=None):
    if attr:
        for val in src:
            list_.append(val[attr])
    else:
        for val in src:
            list_.append(val.get_text())

data = []
# Set Trustpilot page numbers to scrape here
company = args.company[0] # 'www.cheerz.com'

users = []
locations = []
userReviewNum = []
ratings = []

base_url = "https://www.trustpilot.com/review/"

for i in trange(args.from_page, args.to_page + 1):
    try:
        url = base_url + company + "?languages=" + args.language
        if i != 1:
            url += "&page=" + str(i) 

        response = requests.get(url)
        web_page = response.text
        soup = BeautifulSoup(web_page, "html.parser")
        articles = soup.select('article')
        if articles == []:
            break
        for e in articles:
            data.append({
                'Title':e.h2.text,
                'Date': e.select_one('[data-service-review-date-of-experience-typography]').text.split(': ')[-1],
                #'review_rating':e.select_one('[data-service-review-rating] img').get('alt'),
                'Content': e.select_one('[data-service-review-text-typography]').text if e.select_one('[data-service-review-text-typography]') else None,
                'Page':i,
            })

        soup2list(soup.find_all('span', {'class','typography_heading-xxs__QKBS8 typography_appearance-default__AAY17'}), users)
        soup2list(soup.find_all('div', {'class','typography_body-m__xgxZ_ typography_appearance-subtle__8_H2l styles_detailsIcon__Fo_ua'}), locations)
        soup2list(soup.find_all('span', {'class','typography_body-m__xgxZ_ typography_appearance-subtle__8_H2l'}), userReviewNum)
        soup2list(soup.find_all('div', {'class','styles_reviewHeader__iU9Px'}), ratings, attr='data-service-review-rating')

        # To avoid throttling
        sleep(10)
    except:
        pass


review_data = pd.DataFrame(data)

review_data["Username"] = users
review_data["Total reviews"] = userReviewNum
review_data["Location"] = locations
review_data["Rating"] = ratings


review_data.to_excel('data/Trustpilot/'+company + '.xlsx', index=False)
