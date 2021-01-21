"""
Download reviews using urls to films.
"""
import requests
from bs4 import BeautifulSoup
import numpy as np
import time
import os
from tqdm import tqdm

# put headers with browser info, cookies, etc.
headers = {}


def load_data(url):
    """
    Load html pages, get reviews parts from them
    and clean them from service layouts.
    """
    # send http request to obtain page with reviews
    r = requests.get(url, params=headers)
    # create html parser
    soup = BeautifulSoup(r.text, 'html.parser')
    # choose review parts
    reviews = soup.find_all(class_='_reachbanner_')
    reviews_clean = []
    # clean from service html layouts
    for review in reviews:
        reviews_clean.append(review.find_all(text=True))
    return reviews_clean


def convert(reviews):
    """
    Convert reviews to strings.
    """
    review_converted = []
    for review in reviews:
        # convert parts of review to string and join them
        for i in review:
            map(str, i)
        review = ''.join(review)
        review_converted.append(review)
    return review_converted


def get_name(url):
    """
    Get film name from film page (need to save reviews logically).
    """
    # send http request to obtain page with film
    r = requests.get(url, params=headers)
    # create html parser
    soup = BeautifulSoup(r.text, 'html.parser')
    # choose layout where movie name is presented
    name_clean = soup.find('button', id="movie-trailer-button")
    # check that film name was parsed
    if name_clean is not None:
        name_clean = name_clean['data-title']
    else:
        name = str(np.random.randint(1000, 10000))
    return str(name_clean)


def parsing(url, status, path):
    """
    Main function to parse and save reviews for one film
    """
    # number of page with reviews
    page = 1
    # possible delays (pauses) not to be recognized as a bot
    delays = [5, 6, 7, 8, 5.5, 6.5, 7.5, 4.5]
    # get film name
    name = get_name(url)
    # do pause
    time.sleep(np.random.choice(delays))
    # while there are next page with reviews
    while True:
        # url of page with reviews
        loaded_data = load_data(
            url + 'reviews/ord/date/status/{}/perpage/200/page/{}/'.format(status, page))
        # no such page
        if loaded_data == []:
            break
        else:
            # create 3 directories (corresponds with labels)
            if not os.path.exists(path + r'/{}'.format(status)):
                os.makedirs(path + r'/{}'.format(status))
            # process raw reviews
            converted_data = convert(loaded_data)
            # save reviews strings
            for i, review in enumerate(converted_data):
                try:
                    with open(path + r'/{}/{}_{}_{}.txt'.format(status, name, page, i),
                              'w', encoding='utf-8') as output:
                        output.write(review)
                except:
                    print(name, page, i)
            # go to next page
            page += 1
            # do pause
            time.sleep(np.random.choice(delays))


# path where reviews should be saved
path = '/data/hse_bigdata/kliat/kinopoisk_reviews'
# get list of urls with films
with open('urls_list.txt') as f:
    urles = f.readlines()
urles = [x.strip() for x in urles]
# possible labels
statuses = ['good', 'bad', 'neutral']
# possible delays (pauses) not to be recognized as a bot
delays = [5, 6, 7, 8, 5.5, 6.5, 7.5, 4.5]
# parse EACH THIRD film (not all because it is too many)
for url in tqdm(urles[::3]):
    # go separately for each label
    for status in statuses:
        try:
            # parse one film
            parsing(url=url, status=status, path=path)
            time.sleep(np.random.choice(delays))
        except AttributeError as e:
            # exception may be raised if user was banned
            print(e)
            print('You were banned: {}, {}'.format(url, status))
            break
    # go to else-block if everything is OK
    # if loop over statuses was broken, loop over urls also will stop
    else:
        # one url done
        continue
    break
