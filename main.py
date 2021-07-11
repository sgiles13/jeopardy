import requests
import re
from bs4 import BeautifulSoup as BS4
from timeit import default_timer as timer
import humanfriendly

def get_links(url, item_type):
    r = requests.get(url)
    soup = BS4(r.content, features = 'html.parser')
    if item_type == 'season':
        souplist = soup.find('div', {'id': 'content'}).find('table').find_all('a', href=True)
    elif item_type == 'episode':
        souplist = soup.find('div', {'id': 'content'}).find('table').find_all('a', href=True)
    item_list = []
    for item in souplist:
        item_list.append(item['href'])
    return item_list

# initialize URL and folder, record start time
urlroot = r'https://j-archive.com/'
starttime = timer()

# read the page html and use BeautifulSoup to extract the list of data files
r = requests.get(urlroot + 'listseasons.php')
soup = BS4(r.content, features = 'html.parser')
series_soup = soup.find('div', {'id': 'content'}).find('table').find_all('a', href=True)
#series_list = soup.find('div', {'id': 'content'}).find('table').find('tbody').findAll('tr')
series_list = []
for series in series_soup:
    series_list.append(series['href'])
print(series_list)

episode_list = []
for series in series_list:
    episodes = get_links(urlroot + series, 'episode')
    episode_list.append(episodes)
    #for episode in episode_list:
print(episode_list)
endtime = timer()
print('Completed in ' + humanfriendly.format_timespan(endtime-starttime))