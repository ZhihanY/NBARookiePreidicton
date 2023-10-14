import numpy as np
import pandas as pd
import requests 
from bs4 import BeautifulSoup
from unidecode import unidecode

from NBACom_Scraper.tools import stat_parse_decorator, output_dataframe
from BBR_Scraper.seasons import get_standings

## Define some functions for usage

def get_draft_class(year):
      '''
      -- Retrieve draft data from basketball-reference

      year: the draft year
      '''
      r = requests.get(f'https://www.basketball-reference.com/draft/NBA_{year}.html')

      if r.status_code==200:
        soup = BeautifulSoup(r.content, 'lxml')
        table = soup.find('table')
        df = pd.read_html(str(table))[0]

        # get rid of duplicate pick col
        df.drop(['Unnamed: 0_level_0'], inplace=True, axis = 1, level=0)
        df.rename(columns={'Unnamed: 1_level_0': '', 'Pk': 'PICK', 'Unnamed: 2_level_0': '', 'Tm': 'TEAM',
                  'Unnamed: 5_level_0': '', 'Yrs': 'YEARS', 'Totals': 'TOTALS', 'Shooting': 'SHOOTING',
                  'Per Game': 'PER_GAME', 'Advanced': 'ADVANCED', 'Round 1': '', 
                  'Player': 'PLAYER', 'College': 'COLLEGE'}, inplace=True)

        # flatten columns
        df.columns = ['_'.join(x) if x[0] != '' else x[1] for x in df.columns]

        # remove mid-table header rows
        df = df[df['PLAYER'].notna()]
        df = df[~df['PLAYER'].str.contains('Round|Player')]

        return df

@output_dataframe
@stat_parse_decorator
def get_playerbio(season, season_type='Regular Season', mode='PerGame'):
    '''
    -- Retrieve player bio data from stats.nba.com 
    -- Visit nba.com to understand some of the paramterers below

    playtype: string
        ex. Isolation
    season: string
        The nba season to retrieve
    season_type: string
        The type of season to retrieve, default 'Regular Season'
    mode: string 
        The aggregation method to use, ex. 'PerGame'
    '''

    link = 'https://stats.nba.com/stats/leaguedashplayerbiostats?College=&Conference=&Country=&DateFrom=&DateTo=&Division=&DraftPick=&DraftYear=&GameScope=&GameSegment=&Height=&ISTRound=&LastNGames=0&LeagueID=00&Location=&Month=0&OpponentTeamID=0&Outcome=&PORound=0&PerMode='+ mode +\
        '&Period=0&PlayerExperience=&PlayerPosition=&Season=' + season + '&SeasonSegment=&SeasonType=' + season_type + '&ShotClockRange=&StarterBench=&TeamID=0&VsConference=&VsDivision=&Weight='
    
    headers = {
        'Connection': 'keep-alive',
        'Referer': 'https://www.nba.com/',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36 Edg/115.0.1901.200',
        'Accept-Encoding': 'gzip, deflate, br'
    }
    res = requests.get(link, headers=headers).json()
    return res

def season_handle(s):
    return s + '-' + str(int(s[-2:]) + 1)

def undrafted(row, col):
    if row[col] == None:
        return 61
    else:
        return row[col]

# ------ Data Accquisition Phase -------
draft = pd.DataFrame()

for year in ['2018' , '2019', '2020', '2021', '2022']:
    #data = get_draft_class(year)
    season = season_handle(year)
    data = pd.read_excel(season + ' NBA stats.xlsx', sheet_name='Advanced')
    bio = get_playerbio(season)
    data['Player'] = data['Player'].apply(lambda x: unidecode(x))
    res = bio[bio['DRAFT_YEAR'] == year].merge(data[['Player', 'Tm', 'MP', 'BPM', 'VORP']], left_on='PLAYER_NAME', right_on='Player', how='left')
    draft = pd.concat([draft, res])

draft['DRAFT_ROUND'] = draft.apply(undrafted, axis=1, args=('DRAFT_ROUND', ))
draft['DRAFT_NUMBER'] = draft.apply(undrafted, axis=1, args=('DRAFT_NUMBER', ))
draft['Undrafted'] = draft.apply(lambda x: 1 if x['DRAFT_NUMBER'] == 61 else 0, axis=1)
#temp = bio.merge(data[['Player', 'BPM', 'VORP']], left_on='PLAYER_NAME', right_on='Player', how='left')
draft['PLAYER_WEIGHT'] = draft['PLAYER_WEIGHT'].astype(int)
draft['DRAFT_ROUND'] = draft['DRAFT_ROUND'].astype(int)
draft['DRAFT_NUMBER'] = draft['DRAFT_NUMBER'].astype(int)

draft.dropna(inplace=True)
draft.reset_index(drop=True, inplace=True)


# ------ Modeling Phase ------
from sklearn.linear_model import LogisticRegression, RidgeCV, LinearRegression
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, mean_squared_error, r2_score

X_train, X_test, y_train, y_test = train_test_split(draft[['AGE', 'PLAYER_HEIGHT_INCHES', 
                                                           'PLAYER_WEIGHT', 'DRAFT_ROUND', 'DRAFT_NUMBER']], draft['MP'], test_size=0.2, random_state=42)

# -- Linear Regression 
lm = LinearRegression()
pipe = make_pipeline(StandardScaler(), lm)
pipe.fit(X_train, y_train)
#lm.score(X_train, y_train)
lm_pred = pipe.predict(X_test)
print('----------------------------------------')
print('Linear Regression MSE: %.2f' %mean_squared_error(y_test, lm_pred))
print('Linear Regression R^2: %.2f' %r2_score(y_test, lm_pred))

# -- Ridge Regression
rg = RidgeCV()
pipe = make_pipeline(StandardScaler(), rg)
pipe.fit(X_train, y_train)
rg_pred = pipe.predict(X_test)
pipe.score(X_train, y_train)
print('----------------------------------------')
print('Linear Regression MSE: %.2f' %mean_squared_error(y_test.values, rg_pred))
print('Linear Regression R^2: %.2f' %r2_score(y_test.values, rg_pred))
