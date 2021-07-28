import requests
import re
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup as BS4
from timeit import default_timer as timer
import humanfriendly
import pickle
import argparse
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument('--load_episodes', action='store_true',
        help='load episode URLs from text file')
parser.add_argument('--debug', action='store_true',
        help='if in debug mode, only scrapes first 10 episodes')
parser.add_argument('--write_csv', action='store_true',
        help='write final data to CSV format')
opt = parser.parse_args()
print('\nCommand line arguments entered:')
for param in opt.__dict__.keys():
        print(param, '=', opt.__dict__[param])
print('')

def save_episode_list(l):
    with open('episode_list.txt', 'wb') as fp:
        pickle.dump(l, fp)
def load_episode_list(filename):
    with open('episode_list.txt', 'rb') as fp:   # Unpickling
        return pickle.load(fp)

def html2str(soup):
    #Italic text poses an exception to the format. Test for and remove italics.
    s = []
    for x in soup:
        xstr = str(x)
        if '&lt;i&gt;' in xstr:
            xstr = xstr.replace('&lt;i&gt;', '')
            xstr = xstr.replace('&lt;/i&gt;', '')
        else:
            pass
        s.append(xstr)
    return s

def remove_italics(text):
    text = text.replace('<i>','').replace('</i>','')
    return text

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

class EpisodeData(object):
    def __init__(self, url):
        self.url = url
        print('url = ', url)
        try:
            r = requests.get(url)
            self.soup = BS4(r.content, features = 'html.parser')
        except requests.exceptions.ConnectionError: #in case an exception is thrown
            print('Exception thrown! Retrying...')
            s = requests.Session()
            retries = Retry(total=5, backoff_factor=1, status_forcelist=[ 500, 502, 503, 504 ])
            s.mount(url, HTTPAdapter(max_retries=retries))
            r = s.get(url)
            self.soup = BS4(r.content, features = 'html.parser')
        
    def get_categories(self):
        souplist1 = self.soup.find('div', {'id': 'jeopardy_round'}).find('table', {'class': 'round'}).find_all('td', {'class': 'category'})
        results = [x.find('td', {'class': 'category_name'}).text for x in souplist1]
        souplist2 = self.soup.find('div', {'id': 'double_jeopardy_round'}).find('table', {'class': 'round'}).find_all('td', {'class': 'category'})
        results.extend([x.find('td', {'class': 'category_name'}).text for x in souplist2])
        souplist3 = self.soup.find('div', {'id': 'final_jeopardy_round'}).find('table', {'class': 'final_round'}).find('td', {'class': 'category'}).find('td', {'class': 'category_name'}).text
        results.extend([souplist3])
        return results

    def get_clues(self):
        souplist1 = self.soup.find('div', {'id': 'jeopardy_round'}).find('table', {'class': 'round'}).find_all('td', {'class': 'clue'})
        results = [x.find('td', {'class': 'clue_text'}).text for x in souplist1]
        souplist2 = self.soup.find('div', {'id': 'double_jeopardy_round'}).find('table', {'class': 'round'}).find_all('td', {'class': 'clue'})
        results.extend([x.find('td', {'class': 'clue_text'}).text for x in souplist2])
        souplist3 = self.soup.find('div', {'id': 'final_jeopardy_round'}).find('table', {'class': 'final_round'}).find('td', {'class': 'clue'}).find('td', {'class': 'clue_text'}).text
        results.extend([souplist3])
        return results
    
    def get_answers(self):
        souplist1 = self.soup.find('div', {'id': 'jeopardy_round'}).find('table', {'class': 'round'}).find_all('td', {'class': 'clue'})
        souplist1 = html2str(souplist1)
        results = [x.split('correct_response&quot;&gt;')[-1].split('&lt')[0] for x in souplist1]
        souplist2 = self.soup.find('div', {'id': 'double_jeopardy_round'}).find('table', {'class': 'round'}).find_all('td', {'class': 'clue'})
        souplist2 = html2str(souplist2)
        results.extend([x.split('correct_response&quot;&gt;')[-1].split('&lt')[0] for x in souplist2])
        souplist3 = self.soup.find('div', {'id': 'final_jeopardy_round'}).find('table', {'class': 'final_round'}).find('td', {'class': 'category'}).find_all('div')
        for a in souplist3:
            try:
                if re.match('toggle', a['onmouseover']):
                    text = str(a['onmouseover'])
                    if '<i>' in text:
                        text = remove_italics(text)
                    else:
                        pass
                    text = text.split('correct_response\\">')[-1].split('<')[0]
                    text = text.replace('\\', '')
                    results.extend([text])
            except:
                pass
        return results

    def get_responses(self):
        cont1 = self.get_contestants()[0].split(' ')[0] #contestant 1 first name
        cont2 = self.get_contestants()[1].split(' ')[0] #contestant 2 first name
        cont3 = self.get_contestants()[2].split(' ')[0] #contestant 3 first name

        # initialize response arrays based on num_clues
        num_clues = self.get_num_clues()
        print('num_clues = ', num_clues)
        response_matrix = np.zeros((num_clues, 3))
        q_index = 0
        for round in ['jeopardy_round', 'double_jeopardy_round', 'final_jeopardy_round']:
            if round == 'final_jeopardy_round':
                souplist = self.soup.find('div', {'id': round}).find('table', {'class': 'final_round'}).find('td', {'class': 'category'})
                #print('souplist = ', souplist)
                fj_string = souplist.find('div')['onmouseover']
                #print('souplist right = ', fj_string)
                for j, cont in enumerate([cont1, cont2, cont3]):
                    if cont in fj_string:
                        result = fj_string.split('>' + cont)[0].split('<td class=')[-1]
                        if result == '"right"':
                            response_matrix[q_index, j] = 1
                        else:
                            response_matrix[q_index, j] = -1
            else:
                souplist = self.soup.find('div', {'id': round}).find('table', {'class': 'round'}).find_all('td', {'class': 'clue'})
                for x in souplist:
                    responses_table = x.find('div')['onmouseover'].split('<tr>')[-1].split('</tr')[0]
                    #print('responses_table = ', responses_table)
                    # Currently fails for a "Triple Stumper" clue. Need to fix this.
                    for j, cont in enumerate([cont1, cont2, cont3]):
                        if cont in responses_table:
                            result = responses_table.split('>' + cont)[0].split('<td class=')[-1]
                            if result == '"right"':
                                response_matrix[q_index, j] = 1
                            else:
                                response_matrix[q_index, j] = -1
                        else:
                            pass
                    q_index += 1 #keep track of clue number
        #print('response_matrix = ', response_matrix)
        return response_matrix
    def get_dollar_amt(self):
        results = []
        dd_index = []
        for round in ['jeopardy_round', 'double_jeopardy_round']:
            souplist = self.soup.find('div', {'id': round}).find('table', {'class': 'round'}).find_all('td', {'class': 'clue'})
            for x in souplist:
                try:
                    results.append(int(x.find('table', {'class': 'clue_header'}).find(
                        'td', {'class': 'clue_value'}).text.replace('$', '')))
                except:
                    results.append(int(x.find('table', {'class': 'clue_header'}).find(
                        'td', {'class': 'clue_value_daily_double'}).text.replace('DD: ', '').replace('$', '').replace(',','')))
                    dd_index.append(len(results)-1)
                continue
        num_clues = len(results)
        self.dailydouble = self.get_dailydouble(dd_index, num_clues)
        return results
    def get_clue_no(self):
        results = []
        for round in ['jeopardy_round', 'double_jeopardy_round']:
            souplist = self.soup.find('div', {'id': round}).find('table', {'class': 'round'}).find_all('td', {'class': 'clue'})
            for x in souplist:
                if round == 'jeopardy_round':
                    results.append(int(x.find('table', {'class': 'clue_header'}).find(
                    'td', {'class': 'clue_order_number'}).text))
                    num_jr_clues = len(results)
                else:
                    results.append(int(x.find('table', {'class': 'clue_header'}).find(
                    'td', {'class': 'clue_order_number'}).text) + num_jr_clues)                    
        results.append(len(results) + 1) # Final Jeopardy question always comes last
        #print('results = ', results)
        return results
    def get_clue_loc(self):
        xcoord = []
        ycoord = []
        for round in ['jeopardy_round', 'double_jeopardy_round']:
            souplist = self.soup.find('div', {'id': round}).find('table', {'class': 'round'}).find_all('td', {'class': 'clue'})
            #print('souplist = ', souplist)
            for x in souplist:
                if round == 'jeopardy_round':
                    xcoord.append(int(x.find('td', {'class': 'clue_text'})['id'].split('clue_J_')[-1].split('_stuck')[0].split('_')[0]))
                    ycoord.append(int(x.find('td', {'class': 'clue_text'})['id'].split('clue_J_')[-1].split('_stuck')[0].split('_')[-1]))
                else:
                    xcoord.append(int(x.find('td', {'class': 'clue_text'})['id'].split('clue_DJ_')[-1].split('_stuck')[0].split('_')[0]))
                    ycoord.append(int(x.find('td', {'class': 'clue_text'})['id'].split('clue_DJ_')[-1].split('_stuck')[0].split('_')[-1]))
        xcoord.append(0) #final jeopardy
        ycoord.append(0) #final jeopardy             
        return xcoord, ycoord
    def get_contestants(self):
        souplist = self.soup.find('div', {'id': 'contestants'}).find('table', {'id': 'contestants_table'}).find_all('p', {'class': 'contestants'})
        results = [x.find('a').text for x in souplist]
        return results
    def get_show_date(self):
        results = self.soup.head.find('title').text.split('aired ')[-1]
        return datetime.strptime(results, "%Y-%m-%d")
    def get_dailydouble(self, dd_index, num_clues):
        results = [0]*num_clues
        for i in dd_index:
            results[i] = 1
        return results
    def get_num_clues(self):
        num_clues = 0
        for round in ['jeopardy_round', 'double_jeopardy_round']:
            souplist = self.soup.find('div', {'id': round}).find('table', {'class': 'round'}).find_all('td', {'class': 'clue'})
            num_clues += len(souplist)
        return (num_clues + 1) # add FJ clue
    
    def get_contestant_totals(self, clue_value, responses):
        responses_nofj = responses[:(len(responses)-1)] # drop final jeopardy clue since this does not have an assigned clue value
        earnings_matrix = np.transpose(np.transpose(responses_nofj)*np.array(clue_value))
        earnings_matrix_cumsum = np.zeros(earnings_matrix.shape)
        for i in range(3):
            earnings_matrix_cumsum[:,i] = np.cumsum(earnings_matrix[:,i])
        final_scores = self.get_final_totals()
        earnings_matrix_cumsum = np.vstack((earnings_matrix_cumsum, final_scores))
        return earnings_matrix_cumsum

    def get_final_totals(self):
        souplist = self.soup.find('div', {'id': 'final_jeopardy_round'}).find_all('td', {'class': 'score_positive'})
        #print('souplist final scores = ', souplist)
        try:
            cont1_final = int(str(souplist).split('$')[1].split('<')[0].replace(',',''))
            cont2_final = int(str(souplist).split('$')[2].split('<')[0].replace(',',''))
            cont3_final = int(str(souplist).split('$')[3].split('<')[0].replace(',',''))
        except:
            cont1_final = int(str(souplist).split('class="score_positive">')[1].split('<')[0].replace(',',''))
            cont2_final = int(str(souplist).split('class="score_positive">')[2].split('<')[0].replace(',',''))
            cont3_final = int(str(souplist).split('class="score_positive">')[3].split('<')[0].replace(',',''))
        final_scores = np.array([cont1_final, cont2_final, cont3_final])
        return final_scores

def extract(episode_data):
    # Extract 'get' attributes from episode data. Save in tabular form. 
    # Test if 1st clue. If not, append to previous data.

    num_clues = episode_data.get_num_clues()

    show_date = episode_data.get_show_date()
    contestants =  episode_data.get_contestants()
    xcoord, ycoord = episode_data.get_clue_loc()
    clue_number = episode_data.get_clue_no()
    clue_value = episode_data.get_dollar_amt()
    categories = episode_data.get_categories()
    clues =  episode_data.get_clues()
    answers =  episode_data.get_answers()
    dd = episode_data.dailydouble
    #print('dd = ', dd)
    responses = episode_data.get_responses()
    contestant_totals = episode_data.get_contestant_totals(clue_value, responses)

    show_date = [show_date.strftime('%Y-%m-%d')]*num_clues #Duplicate for all clues
    cont1 = [contestants[0]]*num_clues #Duplicate for all clues
    cont2 = [contestants[1]]*num_clues #Duplicate for all clues
    cont3 = [contestants[2]]*num_clues #Duplicate for all clues
    cont1_resp = list(responses[:,0])
    cont2_resp = list(responses[:,1])
    cont3_resp = list(responses[:,2])
    cont1_total = list(contestant_totals[:,0])
    cont2_total = list(contestant_totals[:,1])
    cont3_total = list(contestant_totals[:,2])
    episode_data = [show_date, cont1, cont2, cont3, xcoord, ycoord, clue_number, clue_value, 
    categories, clues, answers, dd, cont1_resp, cont2_resp, cont3_resp, cont1_total, cont2_total, cont3_total]
    #print('episode data = ', episode_data[0])
    return episode_data

def append_data(episode_list, episode_dict, labels):
    for i, value in enumerate(episode_list):
        #print(i, value)
        #print('episode_dict col = ', episode_dict[labels[i]])
        episode_dict[labels[i]].extend(value)
    return episode_dict

def dict2df(dict):
    df = pd.DataFrame.from_dict(dict)
    return df

def df2csv(df):
    df.to_csv('episode_data.csv')

def main():
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
    #print(series_list)

    if opt.load_episodes:
        episode_list = load_episode_list('episodes_list.txt')
        #print(episode_list)
    else:
        episode_list = []
        for series in series_list:
            episodes = get_links(urlroot + series, 'episode')
            episode_list.append(episodes)
        save_episode_list(episode_list)
    if opt.debug:
        series_list = series_list[:2]
        episode_list = [x[:5] for x in episode_list[:2]]
    else:
        pass
    for i in range(len(series_list)):
        series_episodes = episode_list[i]
        for episode in series_episodes:
            print('episode = ', episode)
            episode_data = EpisodeData(episode)
            episode_data = extract(episode_data)
            #if episode_dict in locals():
            try:
                episode_dict = append_data(episode_data, episode_dict, labels)
            #else:
            except UnboundLocalError:
                labels = ['show_date', 'contestant1', 'contestant2', 'contestant3',
                        'x_coord', 'y_coord', 'clue_number', 'clue_value', 
                        'category', 'question', 'answer', 'daily_double', 
                        'cont1_response', 'cont2_response', 'cont3_response',
                        'earnings1', 'earnings2', 'earnings3']
                episode_dict = dict.fromkeys(labels, [])
                for i, col in enumerate(episode_data):
                    episode_dict[labels[i]] = col
                    #episode_dict = append_data(episode_data, episode_dict, labels)

    df = dict2df(episode_dict)
    if opt.write_csv:
        df2csv(df)
    endtime = timer()
    print('Completed in ' + humanfriendly.format_timespan(endtime-starttime))

if __name__ == '__main__':
    main()