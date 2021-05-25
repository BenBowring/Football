#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 23 23:18:41 2021

@author: benjaminbowring
"""

import pandas as pd
import numpy as np
import os
import itertools
import random
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

FOLDER = os.path.expanduser("~/Desktop/Football/")

data = pd.read_csv(FOLDER + 'E0_21.csv', index_col='Date')
data.index = pd.to_datetime(data.index, format='%d/%m/%Y')

 
class Football:
    
    def __init__(self, data):
        
        self.data = data
        self.data.index = pd.to_datetime(data.index, format='%d/%m/%Y')
        
        self.home_labels = ['GF', 'GA', 'Opp']
        self.home_cols = ['FTHG', 'FTAG', 'AwayTeam']
        
        self.away_labels = ['GF', 'GA', 'Opp']
        self.away_cols = ['FTAG', 'FTHG', 'HomeTeam']
        
        self.teams = data['HomeTeam'].unique()

    def get_history(self, date):
        
        self.doi = date
        self.doi_delta = self.doi + pd.Timedelta("1 days")

        self.season = self.data.loc[:self.doi]
        self.season_remaining = self.data.loc[self.doi_delta:]
        self.matchups_remaining = [[x[0], x[1]] for x in zip(self.season_remaining['HomeTeam'], self.season_remaining['AwayTeam'])]
        
        self.dict_goals = {}
        
        for team in self.teams:
        
            home_games = self.season[self.season['HomeTeam'] == team][self.home_cols]
            home_games.columns = self.home_labels
            home_games['H/A'] = 'H'
            
            away_games = self.season[self.season['AwayTeam'] == team][self.away_cols]
            away_games.columns = self.away_labels
            away_games['H/A'] = 'A'
        
            self.dict_goals[team] = pd.concat([home_games, away_games], axis = 0).sort_index()
        
        self.df_total = pd.DataFrame({'Average Goals For': [self.dict_goals[team]['GF'].mean() for team in self.teams],
                               'Average Goals Against': [self.dict_goals[team]['GA'].mean() for team in self.teams],
                               'Predicted Goals For': [np.mean([self.dict_goals[opp]['GA'].mean() for opp in self.dict_goals[team]['Opp']]) for team in self.teams],
                               'Predicted Goals Against': [np.mean([self.dict_goals[opp]['GF'].mean() for opp in self.dict_goals[team]['Opp']]) for team in self.teams]}, 
                              index = self.teams)
        
        self.df_home = pd.DataFrame({'Average Goals For': [self.dict_goals[team]['GF'][self.dict_goals[team]['H/A'] == 'H'].mean() for team in self.teams],
                               'Average Goals Against': [self.dict_goals[team]['GA'][self.dict_goals[team]['H/A'] == 'H'].mean() for team in self.teams],
                               'Predicted Goals For': [np.mean([self.dict_goals[opp]['GA'].mean() for opp in self.dict_goals[team]['Opp'][self.dict_goals[team]['H/A'] == 'H']]) for team in self.teams],
                               'Predicted Goals Against': [np.mean([self.dict_goals[opp]['GF'].mean() for opp in self.dict_goals[team]['Opp'][self.dict_goals[team]['H/A'] == 'H']]) for team in self.teams]}, 
                              index = self.teams)
        
        self.df_away = pd.DataFrame({'Average Goals For': [self.dict_goals[team]['GF'][self.dict_goals[team]['H/A'] == 'A'].mean() for team in self.teams],
                               'Average Goals Against': [self.dict_goals[team]['GA'][self.dict_goals[team]['H/A'] == 'A'].mean() for team in self.teams],
                               'Predicted Goals For': [np.mean([self.dict_goals[opp]['GA'].mean() for opp in self.dict_goals[team]['Opp'][self.dict_goals[team]['H/A'] == 'A']]) for team in self.teams],
                               'Predicted Goals Against': [np.mean([self.dict_goals[opp]['GF'].mean() for opp in self.dict_goals[team]['Opp'][self.dict_goals[team]['H/A'] == 'A']]) for team in self.teams]}, 
                              index = self.teams)
        
        self.df_form = pd.DataFrame({'Goals For': [self.dict_goals[team]['GF'].ewm(alpha=0.5).mean().iloc[-1] for team in self.teams],
                               'Goals Against': [self.dict_goals[team]['GA'].ewm(alpha=0.5).mean().iloc[-1] for team in self.teams]}, 
                              index = self.teams)
        
        self.df_total['Adjusted Goals For'] = self.df_total.apply(lambda x: (x['Average Goals For']/x['Predicted Goals For']) * x['Average Goals For'], axis = 1)
        self.df_total['Adjusted Goals Against'] = self.df_total.apply(lambda x: (x['Average Goals Against']/x['Predicted Goals Against']) * x['Average Goals Against'], axis = 1) 
        
        self.df_home['Adjusted Goals For'] = self.df_home.apply(lambda x: (x['Average Goals For']/x['Predicted Goals For']) * x['Average Goals For'], axis = 1)
        self.df_home['Adjusted Goals Against'] = self.df_home.apply(lambda x: (x['Average Goals Against']/x['Predicted Goals Against']) * x['Average Goals Against'], axis = 1) 
        
        self.df_away['Adjusted Goals For'] = self.df_away.apply(lambda x: (x['Average Goals For']/x['Predicted Goals For']) * x['Average Goals For'], axis = 1)
        self.df_away['Adjusted Goals Against'] = self.df_away.apply(lambda x: (x['Average Goals Against']/x['Predicted Goals Against']) * x['Average Goals Against'], axis = 1) 
        
    
    def get_poiss(self, matchups, home_adv = 0.5, form_adv = 0.5):
        
        away_adv = 1 - home_adv
        anti_form = 1 - form_adv
        
        self.df_adjust = pd.DataFrame({'Home Team': [x[0] for x in matchups],
                                  'Away Team': [x[1] for x in matchups],
                                  'Pred Home Goals For': [home_adv * self.df_home['Adjusted Goals For'].loc[x[0]] + away_adv * self.df_away['Adjusted Goals For'].loc[x[0]] for x in matchups],
                                   'Pred Home Goals Against': [home_adv * self.df_home['Adjusted Goals Against'].loc[x[0]] + away_adv * self.df_away['Adjusted Goals Against'].loc[x[0]] for x in matchups],
                                   'Pred Away Goals For': [home_adv * self.df_away['Adjusted Goals For'].loc[x[1]] + away_adv * self.df_home['Adjusted Goals For'].loc[x[1]] for x in matchups],
                                   'Pred Away Goals Against': [home_adv * self.df_away['Adjusted Goals Against'].loc[x[1]] + away_adv * self.df_home['Adjusted Goals Against'].loc[x[1]] for x in matchups],
                                   'Home Form For': [self.df_form['Goals For'].loc[x[0]] for x in matchups],
                                   'Home Form Against': [self.df_form['Goals Against'].loc[x[0]] for x in matchups],
                                   'Away Form For': [self.df_form['Goals For'].loc[x[1]] for x in matchups],
                                   'Away Form Against': [self.df_form['Goals Against'].loc[x[1]] for x in matchups]},
                                     index = self.season_remaining.index)
        
        df_out = pd.DataFrame({'Home Team': self.df_adjust['Home Team'],
                               'Away Team': self.df_adjust['Away Team'],
                               'Home Goals': self.df_adjust.apply(lambda x: (anti_form * (home_adv * x['Pred Home Goals For'] + away_adv *  x['Pred Away Goals Against'])
                                                                        + form_adv * np.mean(x[['Home Form For', 'Away Form Against']])), axis = 1),
                               
                               'Away Goals': self.df_adjust.apply(lambda x: (anti_form * (home_adv * x['Pred Away Goals For'] + away_adv *  x['Pred Home Goals Against'])
                                                                        + form_adv * np.mean(x[['Away Form For', 'Home Form Against']])), axis = 1)})
                              
        return(df_out)


    def mse_run(self, granular = 0.05):
        
        if len(self.matchups_remaining) > 0:
        
            variables = np.arange(0, 1 + granular, granular)
    
            mse_full = pd.DataFrame(columns = variables, index = variables)
    
            for entry in itertools.product(variables, variables):
                
                home_adv, form_adv = entry[0], entry[1] 
                
                score_pred = self.get_poiss(self.matchups_remaining, home_adv, form_adv)
                
                mse_df = pd.DataFrame({'Home Team': score_pred['Home Team'],
                                       'Away Team': score_pred['Away Team'],
                                       'Actual Home': self.season_remaining['FTHG'],
                                       'Actual Away': self.season_remaining['FTAG'],
                                       'Predicted Home': score_pred['Home Goals'],
                                       'Predicted Away': score_pred['Away Goals']})
                
                y_true = mse_df[['Actual Home', 'Actual Away']]
                y_pred = mse_df[['Predicted Home', 'Predicted Away']]
                
                mse_full.loc[home_adv, form_adv] = mean_squared_error(y_true, y_pred)
            
            mse_full = mse_full.apply(pd.to_numeric)
    
            opt_home = mse_full.min(axis = 1).idxmin()
            opt_form = mse_full.min(axis = 0).idxmin()
        
            return [opt_home, opt_form, mse_full]
        
        else:
            
            print("Season Finished")
            
            return [np.nan, np.nan, np.nan]
    
        

evolution = data.index[250:].unique()

optimal_weights = pd.DataFrame(columns = ['Home Advantage', 'Form Advantage'],
                               index = evolution)

its_alive = Football(data)

surface_mse = {}

for date in evolution:
    
    print(date)
    
    its_alive.get_history(date)
    
    mse_temp = its_alive.mse_run()
    
    optimal_weights.loc[date] = its_alive.mse_run()[0:2]
    surface_mse[date] = mse_temp[2]
    


# =============================================================================
# 3d Plot
# =============================================================================

Y = variables
X = variables
X, Y = np.meshgrid(X, Y)

threedee = plt.figure().gca(projection='3d')
threedee.plot_wireframe(X, Y, mse_full)
threedee.set_xlabel('Form Advantage')
threedee.set_ylabel('Home Advantage')
threedee.set_zlabel('MSE')

plt.show()

# =============================================================================
# Poisson Dist -- Leading up to match outcome matrix
# Probably best to restructure output such that is assigned to team
# =============================================================================

def win_percent(df_out):
    
    df_perc = pd.DataFrame(index = df_out.index, columns = ['Home Win', 'Away Win', 'Draw'])
    
    for match in df_perc.index:
        
        i = df_out.loc[match]['Home Goals']
        j = df_out.loc[match]['Away Goals']
        
        poiss_i = [((i**x) * np.e ** (-i))/np.math.factorial(x) for x in np.arange(0,11,1)]
        poiss_j = [((j**x) * np.e ** (-j))/np.math.factorial(x) for x in np.arange(0,11,1)]
        poiss_ij = pd.concat([pd.Series([x*y for x in poiss_i]) for y in poiss_j], axis = 1)
        
        p_draw = np.diag(poiss_ij).sum()
        p_i = np.tril(poiss_ij).sum() - p_draw
        p_j = np.triu(poiss_ij).sum() - p_draw
    
        p_draw += (1 - p_draw - p_i - p_j)
        
        df_perc.loc[match] = [p_i, p_j, p_draw]
        
    return(df_perc)
    
def match_outcome(df_perc):
    
    df_outcome = pd.DataFrame([np.random.choice(['Home','Away','Draw'], p = df_perc.loc[match]) for match in df_perc.index], index = df_perc.index, columns = ['Outcome'])
    
    return df_outcome

    
def flow():
    
    df_out = get_poiss(matchups)
    df_perc = win_percent(df_out)
    df_outcome = match_outcome(df_perc)
    
    return df_outcome


season = list(itertools.combinations(teams,2)) + list(itertools.combinations(teams[::-1],2))
random.shuffle(season)


