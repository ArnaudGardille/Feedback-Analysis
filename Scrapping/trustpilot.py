from bs4 import BeautifulSoup
import requests
import pandas as pd
from time import sleep

def soup2list(src, list_, attr=None):
    if attr:
        for val in src:
            list_.append(val[attr])
    else:
        for val in src:
            list_.append(val.get_text())

data = []
# Set Trustpilot page numbers to scrape here
from_page = 1
to_page = 471
company = 'www.cheerz.com'

users = []
locations = []
userReviewNum = []
ratings = []

for i in range(from_page, to_page + 1):
    response = requests.get(fr"https://www.trustpilot.com/review/{company}?page={i}")
    web_page = response.text
    soup = BeautifulSoup(web_page, "html.parser")

    for e in soup.select('article'):
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




review_data = pd.DataFrame(data)

review_data["Username"] = users
review_data["Total reviews"] = userReviewNum
review_data["Location"] = locations
review_data["Rating"] = ratings


review_data.to_csv(company + '.csv')
