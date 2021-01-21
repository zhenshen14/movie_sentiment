"""
Script to get list of urls to most popular films (1000 urls).
"""
import requests
from bs4 import BeautifulSoup
import numpy as np
import time
import os

# put headers with browser info, cookies, etc.
headers = {}

# urls to pages with popular films list
root_urls = [
    'https://www.kinopoisk.ru/top/lists/186/filtr/all/sort/order/perpage/200/page/1/',
    'https://www.kinopoisk.ru/top/lists/186/filtr/all/sort/order/perpage/200/page/2/',
    'https://www.kinopoisk.ru/top/lists/186/filtr/all/sort/order/perpage/200/page/3/',
    'https://www.kinopoisk.ru/top/lists/186/filtr/all/sort/order/perpage/200/page/4/',
    'https://www.kinopoisk.ru/top/lists/186/filtr/all/sort/order/perpage/200/page/5/', ]


def get_list_urls():
    """
    Load and parse pages from `root_urls` to obtain films urls.
    """
    film_urls = []
    for root_url in root_urls:
        # pause not to be recognized as a bot
        time.sleep(5)
        # send http request
        r = requests.get(root_url, params=headers)
        # create html parser
        soup = BeautifulSoup(r.text, 'html.parser')
        # save urls from parsed html
        for link in soup.find_all('a'):
            href = link.get('href')
            if href is not None and href.startswith('/film/') and 'cast' not in href:
                film_urls.append('https://www.kinopoisk.ru'+href)
    # one url may exist several times in html
    # so we need to remove repeated ones
    film_urls = np.unique(film_urls)
    return film_urls


# save urls to service file
list_urls = get_list_urls()
with open('urls_list.txt', 'w') as f:
    for u in list_urls:
        f.write(u+'\n')
