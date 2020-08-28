# Python Code to list countries region area wise.
# Author: Charudatta Chousalkar
# Run Script as >>python P2_WebScrapping.py

import requests
from bs4 import BeautifulSoup
import pandas as pd

web_link = "https://en.wikipedia.org/wiki/List_of_Asian_countries_by_area"
link = requests.get(web_link).text
#print(link)

# Beautyfying above link.
soup = BeautifulSoup(link, "lxml")
#print(soup)

print(soup.title.string,"\n") # to remove <title> tag

# Identify right table first based on class
country_table = soup.find('table', class_='wikitable sortable')
#print(country_table)

table_links = country_table.find_all('a')
#print(table_links)

# Countries in list
country = []
for links in table_links:
	country.append(links.get('title'))

# Countries in DataFrame
df_country = pd.DataFrame()
df_country['Country'] = country

print(df_country)