#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 15:35:23 2018

@author: Ian, Prince, Brenton, Alex

This creates a class for training session with the following methods:
    - start()
    - train_operator()
    - get_timestamp()
    - cal_performance()
    - save_session_results()
    - reset_episode_action_history()

"""

import numpy as np
import matplotlib.pyplot as plt
from env import env
from rl_brain import agent
from dqn import DeepQNetwork
import datetime
import os
import sys
import pandas as pd

class trainer():
    
    def __init__(self, station_history):
        
        # Session Properties
        self.episodes = []
        self.stock_type = ""
        self.logging = False
        self.env_debug = False
        self.rl_debug = False
        self.bike_station = None
        self.operator = None
        self.sim_stock = []
        self.model_based = False
        self.ID = None
        self.method = None
        self.station_history = station_history
        
        # Performance Metric
        self.success_ratio = 0
        self.rewards = []  # [[r from session 1], [r from session 2] ...]
        self.avg_rewards = [] #[np.mean([r from session 1]), np.mean([r from session 2])...]
        self.final_stocks = [] # [[stock from session 1], [stock from session 2] ...]
        self.episode_action_history = []
        self.episode_stock_history = []
        self.session_action_history = []
        self.session_stock_history = []
        self.q_tables = []
        #self.actions = [-10, -3, -1, 0]
        self.actions = [-30,-20,-10,0,10,20,30]
        
    
    def start(self, episodes, stock_type, logging, env_debug, rl_debug, brain, ID, model_based, algorithm):
        #brain: which method to use. Q learning vs DQN
        
        self.episodes = episodes
        self.stock_type = stock_type
        self.logging = logging
        self.env_debug = env_debug
        self.rl_debug = rl_debug
        self.brain = brain
        self.ID = ID
        self.model_based = model_based
        self.algorithm = algorithm  # ADDED BY FB
        
        if brain == 'q' and model_based == False:
            self.method = algorithm
        
        idx = 0
        print('Trainning epsoid value: ',  self.episodes)
        for eps in self.episodes:
            #print('Trainning forloop, eps value: ',eps)
        
            # Initiate new evironment and RL agent
            self.bike_station = env(self.stock_type, debug = self.env_debug, ID = self.ID,
                                    station_history = self.station_history, total_EPS = eps, current_EPS = 0)
            self.sim_stock.append(self.bike_station.get_sim_stock())

            if self.brain == 'q':
                self.operator = agent(epsilon = 0.99, lr = 0.01, gamma = 1,
                                  current_stock = self.bike_station.current_stock(),
                                  debug = self.rl_debug,
                                  expected_stock = self.bike_station.get_expected_stock(),
                                  model_based = model_based)
            elif self.brain == 'dqn':
                self.operator = DeepQNetwork(self.bike_station.n_actions, self.bike_station.n_features, 0.01, 0.9)
            else:
                print("Error: pick correct brain")
                break
            
            # Train the RL agent and collect performance stats
            rewards, final_stocks = self.train_operator(idx, len(self.episodes), eps,
            logging = self.logging, brain = self.brain, model_based = self.model_based)
            
            # Log the results from this training session
            self.rewards.append(rewards)
            self.avg_rewards.append(np.mean(rewards))
            self.final_stocks.append(final_stocks)
            self.q_tables.append(self.operator.get_q_table())
            self.session_action_history.append(self.episode_action_history)
            self.session_stock_history.append(self.episode_stock_history)
            self.reset_episode_history()
            
            # Destroy the environment and agent objects
            self.bike_station = None
            self.operator = None
            
            idx += 1
        
        if logging == True:
            if self.brain == 'q':
                self.save_session_results(self.get_timestamp(replace = True))
            else:
                self.save_session_results_dqn(self.get_timestamp(replace = True))
            
        return
    
    
    def train_operator(self, idx, num_sessions, episodes, logging, brain, model_based):
    
        '''
        This function trains an RL agent by interacting with the bike station 
        environment. It also tracks and reports performance stats.
        Input:
            - episodes: a int of episode to be trained in this session (e.g. 500)
        Output:
            - reward_list: a list of reward per episode in this sesison
            - final_stocks: a list of final stocks per episode in this session
        '''
        
        print("Start training the Agent ...")
        rewards = 0
        reward_list = []
        final_stocks = []
        step = 0
        
        for each_eps in range(episodes):
            '''print('from trainning each_eps is: ', each_eps,
                  'range(episodes) is', episodes)'''
            self.bike_station.reset(each_eps)
                
            while True:
                
                # Agent picks an action (number of bikes to move)
                # Agent sends the action to bike station environment
                # Agent gets feedback from the environment (e.g. reward of the action, new bike stock after the action, etc.)
                # Agent "learn" the feedback by updating its Q-Table (state, action, reward)
                # Repeat until end of day (23 hours)
                # Reset bike station environment to start a new day, repeat all
                
                
                if self.brain == 'q':
                    #ADDED BY FB
                    action = self.operator.choose_action(self.bike_station.get_old_stock(),
                                                         self.bike_station.get_current_hour(),
                                                         self.bike_station.get_expected_stock())
                    if self.algorithm == 'qlearning': #ADDED BY FB THIS LINE
                        #DISABLED BY FB
                        #action = self.operator.choose_action(self.bike_station.get_old_stock(),
                        #                                self.bike_station.get_expected_stock())
                        current_hour, old_stock, new_stock, expected_stock, _, reward, done, game_over = \
                            self.bike_station.ping(action, each_eps)
                    # ADDED BY FB - SARSA
                    elif self.algorithm == 'sarsa':
                        current_hour, old_stock, new_stock, expected_stock, old_expected_stock, reward, done, \
                            game_over = self.bike_station.ping(action, each_eps)
                        action_fbn = self.operator.choose_action_fb(new_stock,  self.bike_station.get_current_hour(), old_expected_stock)
                #DISABLED BY FB



                #observation_, reward, done = self.bike_station.ping(action)
                if done == True:

                    msg = ("{} of {} Session | Episode: {} | Final Stock: {} |Final Reward: {:.2f}".format(idx+1,
                          num_sessions, (each_eps+1), old_stock, rewards))

                    sys.stdout.write('\r'+msg)
                    sys.stdout.flush()
                    #print("{} of {} Session | Episode: {} | Final Stock: {} |Final Reward: {:.2f}".format(idx,
                     #     num_sessions, each_eps, old_stock, rewards))

                    
                    reward_list.append(rewards)
                    final_stocks.append(old_stock)
                    rewards = 0
                    
                    # Log session action history by episode
                    if brain == 'q':
                        self.episode_action_history.append(self.operator.get_hourly_actions())
                        self.episode_stock_history.append(self.operator.get_hourly_stocks())
                        self.operator.reset_hourly_history()
                    else:
                        self.episode_stock_history.append(self.operator.get_hourly_stocks());
                        self.operator.reset_hourly_history()
                                    
                    break


                if brain == 'q':
                    if self.algorithm == 'qlearning':  # ADDED BY FB THIS LINE
                        current_hour = self.bike_station.get_current_hour()
                        self.operator.learn(old_stock, current_hour, action, reward, new_stock, expected_stock, game_over)
                    # ADDED BY FB - SARSA
                    elif self.algorithm == 'sarsa':
                        self.operator.learn_fb(old_stock,current_hour, action, action_fbn, reward, new_stock, expected_stock,
                                               game_over)



                step +=1
                rewards += reward
                
                # Log hourly action history by each episode


            '''with open('log.txt', 'a') as f:
                msg = ("{} of {} Session | Episode: {} | Final Stock: {} |Final Reward: {:.2f}".format((idx+1),
                                                                                                       num_sessions,
                                                                                                       (each_eps+1),
                                                                                                       old_stock,
                                                                                                       rewards))
                f.write(msg + '\n')'''

                            
        return reward_list, final_stocks
    
    def get_timestamp(self, replace):
        
        if replace == True:
        
            return str(datetime.datetime.now()).replace(" ", "").replace(":", "").\
                        replace(".", "").replace("-", "")
        
        else:
            
            return str(datetime.datetime.now())
    
    
    def reset_episode_history(self):
        
        self.episode_action_history = []
        self.episode_stock_history = []
        
    
    def cal_performance(self):
        
        successful_stocking = []
        
        print("===== Performance =====")
        
        for session in range(len(self.final_stocks)):
            length = len(self.final_stocks[session])
            num_overstock = np.count_nonzero(np.array(self.final_stocks[session]) > 45)
            num_understock = np.count_nonzero(np.array(self.final_stocks[session]) <= 0)
            ratio = (length - num_understock - num_overstock)*100 / length
            
            print("Session {} | Overstock {} Times | Understock {} Times | {}% Successful".format(session, num_overstock, 
                  num_understock, ratio))

            average_reward = round(self.avg_rewards[session], 2)
            print("Average Episode Reward for Session: {}".format(average_reward))
            
            successful_stocking.append(ratio)
        
        return successful_stocking
    
    
    def save_session_results(self, timestamp):
        
        '''
        This function logs the following: 
            - overall success ratio of each session
            - line chart of success ratio by session
            - line chart of reward history by session
            - Q Table of each session
            - Comparison Line Chart of First and Last Episode Hourly Actions
        '''
        
        # --- create a session folder ---
        dir_path = "./performance_log/" + timestamp
        
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            
        successful_stocking = self.cal_performance()
        
        # --- Write Success Rate to File ---
        fname = dir_path + "/success_rate - " + timestamp + ".txt"
        #print('from trainnig line 291, Q-Table: ', self.q_tables)
        with open(fname, 'w') as f:
            
            f.write("Logged at {}".format(self.get_timestamp(replace = False)))
            f.write("\n")
            f.write("This training session ran episodes: {}".format(self.episodes))
            f.write("\n")
        
            for session in range(len(successful_stocking)):
                f.write("Session {} | Episodes: {} | Success Rate: {:.2f}%".format(session, 
                        self.episodes[session], successful_stocking[session]))
                f.write("\n")
        
        # --- Plot Overall Success Rate by Episode ---
        
        title = "% of Successful Rebalancing - " + timestamp
        
        fig1 = plt.figure()
        plt.plot(self.episodes, successful_stocking)
        plt.xlabel("Episodes")
        plt.ylabel("% Success Rate")
        plt.title(title)
        fig1.savefig(dir_path + "/session_success_rate_" + timestamp)
        
        # --- Plot Reward History by Training Session ---
        '''print ('form trainnig line 321 self.rewards ', self.rewards,
               ' sessions, ',len(self.rewards),
               ' self.episodes[session] ',self.episodes[session])'''

        for session in range(len(self.rewards)):
            df_session_rewards = pd.DataFrame(self.rewards[session])
            df_session_rewards.to_csv(dir_path+'/session_'+str(session)+'_rewards.csv')
           #print ('save session ', session, ' rewards')
            fig = plt.figure(figsize=(10, 8))
            
            title = "Reward History by Training Session " + str(session) + " - " + timestamp
            #print('from training.py saving session: (self.episodes[session]): ', (self.episodes[session]))

            x_axis = [x for x in range(self.episodes[session])]
            #print ('from training.py saving session: x_axis: ', x_axis)
            #print('from training.py saving session: self.rewards[session]: ',self.rewards[session])
            plt.plot(x_axis, self.rewards[session], label = "Session "+str(session))
            plt.legend()
            plt.xlabel("Episode")
            plt.ylabel("Reward")
            plt.title(title)
            fig.savefig(dir_path + "/reward_history_session_" + \
                        str(session) + timestamp)
            
        # --- Plot Average Reward History by Training Session ---
        figR = plt.figure(figsize=[10, 8])
        lengths = [len(r) for r in self.rewards]
        means = [np.mean(r) for r in self.rewards]
        if len(self.rewards) > 1:
            increment = (lengths[1]-lengths[0])/20
        else:
            increment = lengths[0]/20

        for reward_list in self.rewards:
            Q3 = np.percentile(reward_list, 75)
            Q1 = np.percentile(reward_list, 25)
            M = np.mean(reward_list)
            location = len(reward_list)
            plt.plot([location-increment, location+increment], [Q1, Q1], 'k-')
            plt.plot([location-increment, location+increment], [Q3, Q3], 'k-')
            plt.plot([location, location], [Q1, Q3], 'k-')
            plt.scatter(location, M, s=100, color='dodgerblue')           

        plt.xlabel('Number of Episodes in Session')
        plt.ylabel('Average Reward per Episode')
        plt.title('Average Reward vs. Session Size', size=20)
        plt.xticks(lengths)

        plt.plot(lengths, means, linestyle='--')
        
        figR.savefig(dir_path + "/reward_averages")

        # --- Save Q tables --- 
        
        for session in range(len(self.q_tables)):
            
            self.q_tables[session].to_csv(dir_path + "/q_table_session_" + \
                        str(session) + timestamp + ".csv")
        
        # --- Comparison Line Chart of First and Last Episode for each Session ---
        
        file_path = dir_path + "/action_history"
        
        if not os.path.exists(file_path):
            os.makedirs(file_path)       
        
        '''print ('from training line 384, self.session_action_history', self.session_action_history,
               'len(self.session_action_history)',len(self.session_action_history))'''
        for session in range(len(self.session_action_history)):
            df_session_action_history = pd.DataFrame(self.session_action_history[session])
            df_session_action_history.to_csv(dir_path + '/session_' + str(session) + '_action_history.csv')
            '''print('save session ', session, ' action_history')'''

            first_eps_idx = 0
            last_eps_idx = len(self.session_action_history[session])-1
            
            fig = plt.figure(figsize=(10, 8))
            title = "Session " + str(session) + " - Hourly Action of Eps " + str(first_eps_idx) + " and Eps " + str(last_eps_idx)
            
            x_axis = [x for x in range(len(self.session_action_history[session][0]))]
            plt.plot(x_axis, self.session_action_history[session][0], label = "Eps 0")
            plt.plot(x_axis, self.session_action_history[session][-1], 
                     label = "Eps " + str(last_eps_idx))
            
            plt.legend()
            plt.xlabel("Hours")
            plt.ylabel("Number of Bikes Moved")
            plt.title(title)
            
            fig.savefig(file_path + "/action_history_" + str(session) + timestamp)
        
        
        # --- Comparison Line Chart of Simulated and Rebalanced Bike Stock --- #
        file_path = dir_path + "/stock_history"
        
        if not os.path.exists(file_path):
            os.makedirs(file_path)

        '''print ('from training line 413, self.session_stock_history', self.session_stock_history,
               'len(self.session_stock_history)',len(self.session_stock_history))

        print('from training line 416, self.sim_stock', self.sim_stock,
             'len(self.sim_stock)', len(self.sim_stock))'''



        for session in range(len(self.session_stock_history)):

            df_session_stock_history = pd.DataFrame(self.session_stock_history[session])
            df_session_stock_history.to_csv(dir_path + '/session_' + str(session) + '_stock_history.csv')
            #print('save session ', session, ' stock_history')

            df_sim_stock = pd.DataFrame(self.sim_stock[session])
            df_sim_stock.to_csv(dir_path + '/session_' + str(session) + '_sim_stock.csv')
            #print('save session ', session, ' sim_stock')

            first_eps_idx = 0
            last_eps_idx = len(self.session_action_history[session])-1
            
            fig = plt.figure(figsize=(10, 8))
            title = "[" + self.method + "]" + "Session " + str(session) + " - Original vs. Balanced Bike Stock after " + str(first_eps_idx) + " and Eps " + str(last_eps_idx)
            
            x_axis = [x for x in range(len(self.session_stock_history[session][0]))]

            #print ('from trainning around line 400 x_axis, simstock', x_axis, self.sim_stock[session])
            # need to plot an average TQ
            plt.subplot(211)
            title0 = "[" + self.method + "]" + "Session " + str(session) + " - Original vs. Balanced Bike Stock on Episode 0"
            plt.axhline(y=45, c="r", ls="--", label="Upper Stock Limit")
            plt.axhline(y=0, c="r", ls="--", label="Lower Stock Limit")
            plt.plot(x_axis, self.sim_stock[session][:24], label = "Original without Balancing - Eps 0",c="g", ls=':')
            plt.plot(x_axis, self.session_stock_history[session][0], label="Balanced Bike Stock - Eps 0", c="g", ls='-')

            plt.legend()
            plt.xlabel("Hours")
            plt.ylabel("Number of Bike Stock")
            plt.title(title0)
            #
            plt.subplot(212)
            title1 = "[" + self.method + "]" + "Session " + str(session) + " - Original vs. Balanced Bike Stock on Episode " + str(last_eps_idx)
            plt.axhline(y=45, c="r", ls="--", label="Upper Stock Limit")
            plt.axhline(y=0, c="r", ls="--", label="Lower Stock Limit")
            plt.plot(x_axis, self.sim_stock[session][24*last_eps_idx:24*(last_eps_idx+1)],
                     label = "Original without Balancing - Eps " + str(last_eps_idx),c="c", ls=':')
            plt.plot(x_axis, self.session_stock_history[session][-1],
                     label = "Balanced Bike Stock - Eps " + str(last_eps_idx),c="c", ls='-')
            plt.legend()
            plt.xlabel("Hours")
            plt.ylabel("Number of Bike Stock")
            plt.title(title1)
            
            fig.savefig(file_path + "/stock_history_" + str(session) + timestamp)
        
        return

    def save_session_results_dqn(self, timestamp):
        dir_path = "./performance_log/" + timestamp
        
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            
        # --- Comparison Line Chart of Simulated and Rebalaned Bike Stock --- #
        file_path = dir_path + "/stock_history"
        
        if not os.path.exists(file_path):
            os.makedirs(file_path)       
        
        
        successful_stocking = self.cal_performance()
        
        # --- Write Success Rate to File ---
        fname = dir_path + "/success_rate - " + timestamp + ".txt"
        
        with open(fname, 'w') as f:
            
            f.write("Logged at {}".format(self.get_timestamp(replace = False)))
            f.write("\n")
            f.write("This training session ran episodes: {}".format(self.episodes))
            f.write("\n")
        
            for session in range(len(successful_stocking)):
                f.write("Session {} | Episodes: {} | Success Rate: {:.2f}%".format(session, 
                        self.episodes[session], successful_stocking[session]))
                f.write("\n")

            # --- Plot Overall Success Rate by Episode ---
        
        title = "% of Successful Rebalancing - " + timestamp
        
        fig1 = plt.figure()
        plt.plot(self.episodes, successful_stocking)
        plt.xlabel("Episodes")
        plt.ylabel("% Success Rate")
        plt.title(title)
        fig1.savefig(dir_path + "/session_success_rate_" + timestamp)

        for session in range(len(self.session_stock_history)):
            
            first_eps_idx = 0
            last_eps_idx = len(self.session_stock_history[session])-1
            
            fig = plt.figure(figsize=(10, 8))
            title = "[" + self.method + "]" + " Session " + str(session) + " - Original vs. Balanced Bike Stock after " + str(first_eps_idx) + " and Eps " + str(last_eps_idx)
            
            x_axis = [x for x in range(len(self.session_stock_history[session][0]))]
            plt.plot(x_axis, self.sim_stock[session], label = "Original without Balancing")
            plt.plot(x_axis, self.session_stock_history[session][0], label = "Balanced Bike Stock - Eps 0")
            plt.plot(x_axis, self.session_stock_history[session][-1], 
                     label = "Balanced Bike Stock - Eps " + str(last_eps_idx))
            
            plt.axhline(y = 50, c = "r", ls = "--", label = "Upper Stock Limit")
            plt.axhline(y = 0, c = "r", ls = "--", label = "Lower Stock Limit")
            
            plt.legend()
            plt.xlabel("Hours")
            plt.ylabel("Number of Bike Stock")
            plt.title(title)
            
            fig.savefig(file_path + "/stock_history_" + "DQN" + str(session) + timestamp)
        
        return
    
    
