# create a txt file include url list of valid text data
  

import urllib
from bs4 import BeautifulSoup
import re
import requests
from urllib.request import urlopen

url = 'https://www.mercurynews.com/'

html = urlopen(url)
soup = BeautifulSoup(html, 'html.parser')
tags = soup('a')
target = open("/Users/lucui/NLP_projects/20200602_web_doc/data/url.txt", 'w')
tags = list(set(tags))

counter = 0

for tag in tags:
        url = tag.get('href', None)
        # index = tags.index(tag)
        # print("index =", index, ":", "url =",url)
        if url and re.match(r"^\w+",url) and url[:28] == "https://www.mercurynews.com/":
                target.write(tag.get('href', None)[0:] + "\n")
                counter += 1
        else:
                continue

print("counter =", counter)
target.close()
~               
