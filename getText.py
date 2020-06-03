# batch data extraction

import urllib
import requests
from urllib.request import urlopen
from bs4 import BeautifulSoup
import re
from urllib.parse import urlparse
import pandas as pd 


url_file = open("/Users/lucui/NLP_projects/20200602_web_doc/data/url.txt")


url_index_list = []
url_out = []
url_content_out = []


i = 0
for url in url_file:

    print("url_index", i)

    if i >= 10000:
        break
    else:

        result = urlparse(url)
        html = urlopen(url).read()
        html_text = requests.get(url)
        pid = re.findall(r"\b\d+\b",url)

        soup = BeautifulSoup(html, 'html.parser')
        infile = soup.get_text()

        copy = False

        # begin = infile.rfind("He",)
        # end = infile.rfind("",)
        
        tags = soup('meta')    
        
        dp_out = []
        for tag in tags:
            dp = tag.get('content', None)

            if dp and len(dp) > 160:
                dp_out.append(dp)
            dp_out_concat = " ".join(dp_out)

        # infile = str(infile[begin:end-1].encode('utf-8').strip())
        infile = str(infile.encode('utf-8').strip())


    if dp_out:
        url_content_out.append(dp_out_concat)

        url_out.append(url)

        url_index_list.append(i)
        i += 1




df = pd.DataFrame(list(zip(url_index_list, url_out, url_content_out)), columns =['id', 'url', 'text']) 
df.to_csv("/Users/lucui/NLP_projects/20200602_web_doc/data/url_text_w_id.csv", header=True, index=False, sep="|")
