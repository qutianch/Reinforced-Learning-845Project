#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  3 12:26:48 2018

@author: Ian

This script is for creating a RL agent class object. This object has the 
following method:
    
    1) choose_action: this choose an action based on Q(s,a) and greedy eps
    2) learn: this updates the Q(s,a) table
    3) check_if_state_exist: this check if a state exist based on env feedback

"""

import numpy as np
import pandas as pd

class agent():
    
    
    def __init__(self, epsilon, lr, gamma, current_stock, debug, expected_stock, model_based):
        
        print("Created an Agent ...")
        #self.actions = [-10, -3, -1, 0]
        self.actions = [-30,-20,-10,0,10,20,30]
        self.reward = 0
        self.epsilon = epsilon
        self.lr = lr
        self.gamma = gamma
        self.debug = debug
        self.current_stock = current_stock
        self.expected_stock = expected_stock
        self.model_based = model_based
        
        # performance metric create multiindex dateframe
        my_index = pd.MultiIndex(levels=[[], []],
                                 codes=[[], []],
                                 names=[u'stock', u'hour'])
        self.q_table = pd.DataFrame(index=my_index, columns = self.actions, dtype = np.float64)
        #self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)
        self.hourly_action_history = []
        self.hourly_stock_history = []
       
    def choose_action(self, s,t, ex):
        
        '''
        This funciton choose an action based on Q Table. It also does 
        validation to ensure stock will not be negative after moving bikes.
        Input: 
            - s: current bike stock
            - t: current hour
            - ex: expected bike stock in subsequent hour (based on random forests prediction)
        
        Output:
            - action: number of bikes to move
        
        '''
        
        self.check_state_exist(s,t)
        self.current_stock = s
        self.current_hour = t
        self.expected_stock = ex
        '''print ('from rl_brain, currentStock: ', self.current_stock,
               ' current_hour: ',self.current_hour)'''
        
        # find valid action based on current stock 
        # cannot pick an action that lead to negative stock
        
        # !!!! remove action validation; only rely on reward/penalty !!!
        '''try:
            valid_state_action = self.find_valid_action(self.q_table.loc[(s,t), :])
        except Exception as e:
            pass'''
        if self.model_based == True:
            #Take an average of current stock and expected stock
            try:
                avg = int(round(0.5*s + 0.5*ex))
            except:
                avg = s
            self.check_state_exist(avg)
            valid_state_action = self.q_table.loc[avg, :]

        elif self.model_based == False:
            #valid_state_action = self.q_table.loc[(s,t), :]
            try:
                valid_state_action = self.find_valid_action(self.q_table.loc[(s, t), :]) # trainig wheels on
                #valid_state_action = self.q_table.loc[(s, t), :] #trainig wheels off
            except Exception as e:
                pass
        if np.random.uniform() < self.epsilon:
                        
            try:
                # find the action with the highest expected reward
                
                valid_state_action = valid_state_action.reindex(np.random.permutation(valid_state_action.index))
                action = valid_state_action.astype(float).idxmax()
            
            except:
                # if action list is null, default to 0
                action = 0
                        
            if self.debug == True:
                print("Decided to Move: {}".format(action))
                        
        else:
            
            # randomly choose an action
            # re-pick if the action leads to negative stock
            try:
                action = np.random.choice(valid_state_action.index)
            except:
                action = 0
            
            if self.debug == True:
                print("Randomly Move: {}".format(action))
        
        self.hourly_action_history.append(action)
        self.hourly_stock_history.append(s)
        
        return action

    def learn(self, s, t, a, r, s_, ex, g):

        
        '''
        This function updates Q tables after each interaction with the
        environment.
        Input: 
            - s: current bike stockc
            - ex: expected bike stock in next hour
            - a: current action (number of bikes to move)
            - r: reward received from current state
            - s_: new bike stock based on bike moved and new stock
        Output: None
        '''
        
        if self.debug == True:
            print('rl_brain learn Debug info-----')
            print("current bike stock: {}".format(s))
            print("current hour: {}".format(t))
            print("expected bike stock in next hour: {}".format(ex))
            print("current action (number of bikes to move): {}".format(a))
            print("reward received from current state: {}".format(r))
            print("new bike stock based on bike moved and new stock: {}".format(s_))
            print("---")
        
        self.check_state_exist(s_,t)

        if self.model_based == False:
            '''print ('from RL_brain line 145, qtable is\n', self.q_table,
                   'current stock s is ',s,
                   'current time t is ',t,
                   'current action a is ',a)'''
            q_predict = self.q_table.loc[(s,t-1), a]
        elif self.model_based == True:
            avg = int(round(0.5*s + 0.5*ex))
            self.check_state_exist(avg)
            q_predict = self.q_table.loc[avg, a]
        

        if g == False:
            

            # Updated Q Target Value if it is not end of day
            '''print ('from rl_brain line 160, q_table is,', self.q_table,
                   's_ is ', s_,
                   't is ', t)'''
            q_target = r + self.gamma * self.q_table.loc[(s_,t), :].max()
        
        else:
            # Update Q Target Value as Immediate reward if end of day
            q_target = r

        if self.model_based == False:
            self.q_table.loc[(s,t-1), a] += self.lr * (q_target - q_predict)
        elif self.model_based == True:
            self.q_table.loc[avg, a] += self.lr * (q_target - q_predict)
        
        return

    def choose_action_fb(self, s,t, ex):

        '''
        This funciton_FB SARSA choose an action based on Q Table. It also does
        validation to ensure stock will not be negative after moving bikes.
        Input:
            - s: current bike stock
            - ex: expected bike stock in subsequent hour (based on random forests prediction)

        Output:
            - action: number of bikes to move

        '''

        self.check_state_exist(s,t)
        self.current_stock = s
        self.current_hour = t
        self.expected_stock = ex

        # find valid action based on current stock
        # cannot pick an action that lead to negative stock

        # !!!! remove action validation; only rely on reward/penalty !!!
        # valid_state_action = self.find_valid_action(self.q_table.loc[s, :])
        if self.model_based == True:
            # Take an average of current stock and expected stock
            try:
                avg = int(round(0.5 * s + 0.5 * ex))
            except:
                avg = s
            self.check_state_exist(avg)
            valid_state_action = self.q_table.loc[avg, :]

        elif self.model_based == False:
            #valid_state_action = self.q_table.loc[s, :]
            try:
                valid_state_action = self.find_valid_action(self.q_table.loc[(s, t), :]) # training wheels on for SARSA
                #valid_state_action = self.q_table.loc[(s, t), :]  # trainig wheels off
            except Exception as e:
                pass

        if np.random.uniform() < self.epsilon:

            try:
                # find the action with the highest expected reward

                valid_state_action = valid_state_action.reindex(np.random.permutation(valid_state_action.index))
                action = valid_state_action.astype(float).idxmax()

            except:
                # if action list is null, default to 0
                action = 0

            if self.debug == True:
                print("Decided to Move: {}".format(action))

        else:

            # randomly choose an action
            # re-pick if the action leads to negative stock
            try:
                action = np.random.choice(valid_state_action.index)
            except:
                action = 0

            if self.debug == True:
                print("Randomly Move: {}".format(action))

        #DISABLED BY FB
        #self.hourly_action_history.append(action)
        #self.hourly_stock_history.append(s)

        return action

    def learn_fb(self, s, t, a, a_fbn, r, s_, ex, g):

        if self.debug == True:
            print("Moved Bikes: {}".format(a))
            print("Old Bike Stock: {}".format(s))
            print("New Bike Stock: {}".format(s_))
            print("---")

        self.check_state_exist(s_,t)

        if self.model_based == False:
            q_predict = self.q_table.loc[(s,t-1), a]
        elif self.model_based == True:
            avg = int(round(0.5 * s + 0.5 * ex))
            self.check_state_exist(avg)
            q_predict = self.q_table.loc[avg, a]

        if g == False:

            # Updated Q Target Value if it is not end of day
            q_target = r + self.gamma * self.q_table.loc[(s_,t), a_fbn]  #ADDED BY FB: THIS IS NOW SARSA ALGO

        else:
            # Update Q Target Value as Immediate reward if end of day
            q_target = r

        if self.model_based == False:
            self.q_table.loc[(s,t-1), a] += self.lr * (q_target - q_predict)
        elif self.model_based == True:
            self.q_table.loc[avg, a] += self.lr * (q_target - q_predict)

        return

    def check_state_exist(self, state,current_hour):
        
        # Add a new row with state value as index if not exist
        
        '''if state not in self.q_table.index:

            self.q_table = self.q_table.append(
                pd.Series(
                        [0]*len(self.actions),
                        index = self.q_table.columns,
                        name = state
                        )
                )
        '''
        if not self.q_table.index.isin([(state, current_hour)]).any():
            self.q_table.loc[(state, current_hour), :] = 0

        return


    def find_valid_action(self, state_action):

        '''
        This function check the validity acitons in a given state.
        Input: 
            - state_action: the current state under consideration
        Output:
            - state_action: a pandas Series with only the valid actions that
                            will not cause negative stock
        '''
        
        # remove action that will stock to be out side of the range
        
        for action in self.actions:

            if (self.current_stock + action < 0 or self.current_stock + action > 45):
                
                if self.debug == True:
                    print("Drop action {}, current stock {}".format(action, self.current_stock))
                
                state_action.drop(index = action, inplace = True)

        #print ('from rl_brain line 325, state_action is, ', state_action)
        return state_action
    
    def print_q_table(self):
        
        print(self.q_table)


    def get_q_table(self):
        
        return self.q_table

    
    def get_hourly_actions(self):
        
        return self.hourly_action_history
    
    def get_hourly_stocks(self):
        
        return self.hourly_stock_history

    
    def reset_hourly_history(self):
        
        self.hourly_action_history = []
        self.hourly_stock_history = []
