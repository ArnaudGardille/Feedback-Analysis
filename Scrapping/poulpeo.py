from time import sleep
import requests

import pandas as pd
from bs4 import BeautifulSoup
from tqdm import trange

def soup2list(src, list_, attr=None):
    if attr:
        for val in src:
            list_.append(val[attr])
    else:
        for val in src:
            list_.append(val.get_text())

users = []
userReviewNum = []
ratings = []
locations = []
dates = []
reviews = []

company = 'Cheerz'


result = requests.get(fr"https://www.poulpeo.com/avis/cheerz-htm")
soup = BeautifulSoup(result.content, features="lxml")

# Extract user reviews from each list item
for review in soup.find_all('div', class_='list-item'):
    user = review.find('span', class_='user-name').get_text().strip()
    rating = review.find('span', class_='star-ratings').get_text().strip()
    date = review.find('span', class_='date').get_text().strip()
    content = review.find('div', class_='review-content').get_text().strip()

    # Append extracted info to respective lists
    users.append(user)
    ratings.append(rating)
    dates.append(date)
    reviews.append(content)


review_data = pd.DataFrame({
    'Username': users,
    'Rating': ratings,
    'Date': dates,
    'Content': reviews
})

review_data.to_csv(company + 'poulpeo.csv')
