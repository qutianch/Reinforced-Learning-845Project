#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  3 12:26:58 2018

@author: Ian, Prince, Brenton, Alex

This script is for creating an Environment class. Each environment represents
a bike station with the following methods:
    1) generate: this initialize bike station with stock characteristics
    2) ping: this communicates with RL Agent with current stock info, reward,
                and episode termination status; iterate to new hour
    3) update: this updates the bike stock based on RL Agent Action
    4) reset: reset all environment properties for new episode training

"""

import numpy as np
import pandas as pd
import json
import scipy.stats as stats
from itertools import chain
import random

with open('EXPECTED_BALANCES.json') as json_data:
    expected_balance = json.load(json_data)


weekends_3M_df = pd.read_excel(open('station_stats.xlsx', 'rb'),sheet_name='3_month_weekends')
weekdays_3M_df = pd.read_excel(open('station_stats.xlsx', 'rb'),sheet_name='3_month_weekday')
combined_3M_df = pd.read_excel(open('station_stats.xlsx', 'rb'),sheet_name='3_month_combined')


class env():
    
    def __init__(self, mode, debug, ID, station_history, total_EPS, current_EPS):
        
        print("\nCreating A Bike Environment...")
        
        self.mode = mode
        self.seed = np.random.random_integers(0, 10)
        self.num_hours = 23
        self.current_hour = 0

        if mode == 'combined':
            self.bike_stock_sim = self.generate_stock(mode, ID, total_EPS)

        else:
            self.bike_stock_sim = station_history
            
        self.bike_stock = self.bike_stock_sim.copy() # to be reset to original copy every episode
        self.old_stock = self.bike_stock[0]
        self.new_stock = 0
        self.done = False
        self.reward = 0
        self.bike_moved = 0
        self.debug = debug
        self.ID = str(ID)
        self.total_EPS = total_EPS
        self.current_EPS = current_EPS
        #print('from env total EPS ', self.total_EPS,
        #      'from env current EPS ', self.current_EPS,)
        #exp_bike_stock is list of expected balances in next hour
        #Predictions based on Random Forests model
        self.exp_bike_stock_sim = self.bike_stock_sim.copy()
        self.exp_bike_stock = self.exp_bike_stock_sim.copy()
        #print ('From env line 66, self.exp_bike_stock: ',self.exp_bike_stock)
        self.expected_stock = self.exp_bike_stock[0]
        #print('expected_stock: ', self.expected_stock)
        self.expected_stock_new = 0

        #self.actions = [-10, -3, -1, 0]
        self.actions = [-30,-20,-10,0,10,20,30]
        self.n_actions = len(self.actions)
        #features of the observation: hour, old stock, new stock
        self.n_features = 1

        self.citibike_df = 0
        self.game_over = False

        
        if self.debug == True:
            print("Generating Bike Stock: {}".format(self.mode))
            print("Bike Stock: {}".format(self.bike_stock))
        
    def generate_stock(self, mode, ID, total_EPS):
        
        # generate a list of 24 hourly bike stock based on mode
        # mode: linear, random, real based on citiBike Data
        
        bike_stock = [43]
        
        if mode == "linear":
            for i in range(1, 24):
                bike_stock.append(bike_stock[i-1]+3)
            print('Linear Bike Stock: ',bike_stock)
                
        if mode == "random":
            for i in range(1, 24):
                #bike_stock.append(bike_stock[i-1] + 3 + np.random.random_integers(-5, 5))
                bike_stock.append(bike_stock[i - 1] + 3 + np.random.random_integers(-5, 5))
            print('Random Bike Stock: ', bike_stock)
                
        if mode == "actual":
            pass

        if mode == "weekends":# put function to generate 24hour* num days with means and stdev
            #print('total_eps: ',total_EPS)


            def generateRandomNets(start_mu, start_sigma, start_range_min, start_range_max, end_mu, end_sigma,
                                   end_range_min, end_range_max, num_episodes):
                # print ('\n line: ',start_mu,start_sigma,start_range_min,start_range_max,end_mu,end_sigma,end_range_min,end_range_max,num_episodes)

                start_isSamerange = start_range_min == start_range_max
                end_isSamerange = end_range_min == end_range_max

                # print('start_isSamerange ', start_isSamerange)
                # print('end_isSamerange ', end_isSamerange)

                if (start_isSamerange == True and end_isSamerange == True):
                    # print ('state 1')
                    leaving_stock = [start_range_min] * num_episodes
                    comming_stock = [end_range_min] * num_episodes
                if (start_isSamerange == True and end_isSamerange == False):
                    # print ('state 2')
                    leaving_stock = [start_range_min] * num_episodes
                    end_dist = stats.truncnorm((end_range_min - end_mu) / end_sigma,
                                               (end_range_max - end_mu) / end_sigma, loc=end_mu, scale=end_sigma)
                    comming_stock = end_dist.rvs(num_episodes)
                if (start_isSamerange == False and end_isSamerange == True):
                    # print ('state 3')
                    start_dist = stats.truncnorm((start_range_min - start_mu) / start_sigma,
                                                 (start_range_max - start_mu) / start_sigma, loc=start_mu,
                                                 scale=start_sigma)
                    leaving_stock = start_dist.rvs(num_episodes)
                    comming_stock = [end_range_min] * num_episodes
                if (start_isSamerange == False and end_isSamerange == False):
                    # print ('state 4')
                    start_dist = stats.truncnorm((start_range_min - start_mu) / start_sigma,
                                                 (start_range_max - start_mu) / start_sigma, loc=start_mu,
                                                 scale=start_sigma)
                    end_dist = stats.truncnorm((end_range_min - end_mu) / end_sigma,
                                               (end_range_max - end_mu) / end_sigma, loc=end_mu, scale=end_sigma)
                    # start is the num of bikes leaving and end is the num of bokes ccomming
                    leaving_stock = start_dist.rvs(num_episodes)
                    comming_stock = end_dist.rvs(num_episodes)

                net_stock = []
                zip_object = zip(comming_stock, leaving_stock)

                # print ('comming Stock: ',comming_stock)
                # print ('leaving Stock Stock: ',leaving_stock)
                for list1_i, list2_i in zip_object:
                    net_stock.append(int(list1_i - list2_i))

                #baseBike_count = 20
                #net_stock_withBase = [x + baseBike_count for x in net_stock]
                #return net_stock_withBase
                return net_stock

            hour_net_list = []
            for index, row in weekends_3M_df.iterrows():
                start_mu = row.mean_start
                start_sigma = row.stddev_start
                start_range_min = row.range_min_start
                start_range_max = row.range_max_start
                end_mu = row.mean_end
                end_sigma = row.stddev_end
                end_range_min = row.range_min_end
                end_range_max = row.range_max_end
                num_episodes = total_EPS

                hour_net = generateRandomNets(start_mu, start_sigma, start_range_min, start_range_max, end_mu,
                                              end_sigma, end_range_min, end_range_max, num_episodes)
                hour_net_list.append(hour_net)
                # print(hour_net)
            hour_episode_list = list(map(list, zip(*hour_net_list)))
            hour_episode_list_flat = list(chain.from_iterable(hour_episode_list))

            c = list((np.array_split(hour_episode_list_flat, num_episodes)))

            bike_stock = []
            for each_sublist in c:
                print(each_sublist)
                a = each_sublist
                each_sublist_cumSum = [sum(a[0:x + 1]) for x in range(0, len(a))]
                for i in each_sublist_cumSum:
                    bike_stock.append(i)

            baseBike_count = 43
            bike_stock_withBase = [x + baseBike_count for x in bike_stock]
            bike_stock = bike_stock_withBase
            #print('weekdays Bike Stock: ', bike_stock_withBase)

            #return bike_stock_withBase

        if mode == "weekdays":
            #print('total_eps: ', total_EPS)

            def generateRandomNets(start_mu, start_sigma, start_range_min, start_range_max, end_mu, end_sigma,
                                   end_range_min, end_range_max, num_episodes):
                # print ('\n line: ',start_mu,start_sigma,start_range_min,start_range_max,end_mu,end_sigma,end_range_min,end_range_max,num_episodes)

                start_isSamerange = start_range_min == start_range_max
                end_isSamerange = end_range_min == end_range_max

                # print('start_isSamerange ', start_isSamerange)
                # print('end_isSamerange ', end_isSamerange)

                if (start_isSamerange == True and end_isSamerange == True):
                    # print ('state 1')
                    leaving_stock = [start_range_min] * num_episodes
                    comming_stock = [end_range_min] * num_episodes
                if (start_isSamerange == True and end_isSamerange == False):
                    # print ('state 2')
                    leaving_stock = [start_range_min] * num_episodes
                    end_dist = stats.truncnorm((end_range_min - end_mu) / end_sigma,
                                               (end_range_max - end_mu) / end_sigma, loc=end_mu, scale=end_sigma)
                    comming_stock = end_dist.rvs(num_episodes)
                if (start_isSamerange == False and end_isSamerange == True):
                    # print ('state 3')
                    start_dist = stats.truncnorm((start_range_min - start_mu) / start_sigma,
                                                 (start_range_max - start_mu) / start_sigma, loc=start_mu,
                                                 scale=start_sigma)
                    leaving_stock = start_dist.rvs(num_episodes)
                    comming_stock = [end_range_min] * num_episodes
                if (start_isSamerange == False and end_isSamerange == False):
                    # print ('state 4')
                    start_dist = stats.truncnorm((start_range_min - start_mu) / start_sigma,
                                                 (start_range_max - start_mu) / start_sigma, loc=start_mu,
                                                 scale=start_sigma)
                    end_dist = stats.truncnorm((end_range_min - end_mu) / end_sigma,
                                               (end_range_max - end_mu) / end_sigma, loc=end_mu, scale=end_sigma)
                    # start is the num of bikes leaving and end is the num of bokes ccomming
                    leaving_stock = start_dist.rvs(num_episodes)
                    comming_stock = end_dist.rvs(num_episodes)

                net_stock = []
                zip_object = zip(comming_stock, leaving_stock)

                # print ('comming Stock: ',comming_stock)
                # print ('leaving Stock Stock: ',leaving_stock)
                for list1_i, list2_i in zip_object:
                    net_stock.append(int(list1_i - list2_i))

                '''baseBike_count = 20
                net_stock_withBase = [x + baseBike_count for x in net_stock]
                return net_stock_withBase'''
                return net_stock

            hour_net_list = []
            for index, row in weekdays_3M_df.iterrows():
                start_mu = row.mean_start
                start_sigma = row.stddev_start
                start_range_min = row.range_min_start
                start_range_max = row.range_max_start
                end_mu = row.mean_end
                end_sigma = row.stddev_end
                end_range_min = row.range_min_end
                end_range_max = row.range_max_end
                num_episodes = total_EPS

                hour_net = generateRandomNets(start_mu, start_sigma, start_range_min, start_range_max, end_mu,
                                              end_sigma, end_range_min, end_range_max, num_episodes)
                hour_net_list.append(hour_net)
                # print(hour_net)
            hour_episode_list = list(map(list, zip(*hour_net_list)))
            hour_episode_list_flat = list(chain.from_iterable(hour_episode_list))

            c = list((np.array_split(hour_episode_list_flat, num_episodes)))

            bike_stock = []
            for each_sublist in c:
                #print(each_sublist)
                a = each_sublist
                each_sublist_cumSum = [sum(a[0:x + 1]) for x in range(0, len(a))]
                for i in each_sublist_cumSum:
                    bike_stock.append(i)

            baseBike_count = 43
            bike_stock_withBase = [x + baseBike_count for x in bike_stock]
            #print('weekdays Bike Stock: ', bike_stock_withBase)
            bike_stock = bike_stock_withBase

        if mode == "combined":
            #print('total_eps: ', total_EPS)

            def generateRandomNets(start_mu, start_sigma, start_range_min, start_range_max, end_mu, end_sigma,
                                   end_range_min, end_range_max, num_episodes):
                # print ('\n line: ',start_mu,start_sigma,start_range_min,start_range_max,end_mu,end_sigma,end_range_min,end_range_max,num_episodes)

                start_isSamerange = start_range_min == start_range_max
                end_isSamerange = end_range_min == end_range_max

                # print('start_isSamerange ', start_isSamerange)
                # print('end_isSamerange ', end_isSamerange)

                if (start_isSamerange == True and end_isSamerange == True):
                    # print ('state 1')
                    leaving_stock = [start_range_min] * num_episodes
                    comming_stock = [end_range_min] * num_episodes
                if (start_isSamerange == True and end_isSamerange == False):
                    # print ('state 2')
                    leaving_stock = [start_range_min] * num_episodes
                    end_dist = stats.truncnorm((end_range_min - end_mu) / end_sigma,
                                               (end_range_max - end_mu) / end_sigma, loc=end_mu, scale=end_sigma)
                    comming_stock = end_dist.rvs(num_episodes)
                if (start_isSamerange == False and end_isSamerange == True):
                    # print ('state 3')
                    start_dist = stats.truncnorm((start_range_min - start_mu) / start_sigma,
                                                 (start_range_max - start_mu) / start_sigma, loc=start_mu,
                                                 scale=start_sigma)
                    leaving_stock = start_dist.rvs(num_episodes)
                    comming_stock = [end_range_min] * num_episodes
                if (start_isSamerange == False and end_isSamerange == False):
                    # print ('state 4')
                    start_dist = stats.truncnorm((start_range_min - start_mu) / start_sigma,
                                                 (start_range_max - start_mu) / start_sigma, loc=start_mu,
                                                 scale=start_sigma)
                    end_dist = stats.truncnorm((end_range_min - end_mu) / end_sigma,
                                               (end_range_max - end_mu) / end_sigma, loc=end_mu, scale=end_sigma)
                    # start is the num of bikes leaving and end is the num of bokes ccomming
                    leaving_stock = start_dist.rvs(num_episodes)
                    comming_stock = end_dist.rvs(num_episodes)

                net_stock = []
                zip_object = zip(comming_stock, leaving_stock)

                # print ('comming Stock: ',comming_stock)
                # print ('leaving Stock Stock: ',leaving_stock)
                for list1_i, list2_i in zip_object:
                    net_stock.append(int(list1_i - list2_i))

                '''baseBike_count = 20
                net_stock_withBase = [x + baseBike_count for x in net_stock]
                return net_stock_withBase'''
                return net_stock

            hour_net_list = []
            for index, row in combined_3M_df.iterrows():
                start_mu = row.mean_start
                start_sigma = row.stddev_start
                start_range_min = row.range_min_start
                start_range_max = row.range_max_start
                end_mu = row.mean_end
                end_sigma = row.stddev_end
                end_range_min = row.range_min_end
                end_range_max = row.range_max_end
                num_episodes = total_EPS

                hour_net = generateRandomNets(start_mu, start_sigma, start_range_min, start_range_max, end_mu,
                                              end_sigma, end_range_min, end_range_max, num_episodes)
                hour_net_list.append(hour_net)
                # print(hour_net)
            hour_episode_list = list(map(list, zip(*hour_net_list)))
            hour_episode_list_flat = list(chain.from_iterable(hour_episode_list))

            c = list((np.array_split(hour_episode_list_flat, num_episodes)))

            bike_stock = []
            for each_sublist in c:
                #print(each_sublist)
                a = each_sublist
                each_sublist_cumSum = [sum(a[0:x + 1]) for x in range(0, len(a))]
                for i in each_sublist_cumSum:
                    bike_stock.append(i)

            baseBike_count = 20
            bike_stock_withBase = [x + baseBike_count for x in bike_stock]
            #print('weekdays Bike Stock: ', bike_stock_withBase)
            bike_stock = bike_stock_withBase

        if mode == "test":
            #print('total_eps: ', total_EPS)
            # this generate randomly plus and minus 3  or 0 for the stock
            stock_history = []
            start_stock = 20

            for each_episode in range(total_EPS):
                for i in range(0, 24):
                    if i == 0:
                        start_stock = 20
                        stock_history.append(start_stock)
                    else:

                        random_move = (random.sample(set([3,-3]), 1))[0]
                        next_stock = start_stock + random_move
                        stock_history.append(next_stock)
                        start_stock = next_stock

            bike_stock = stock_history

        return bike_stock



    def ping(self, action,each_eps):
        
        # share back t+1 stock, reward of t, and termination status
        self.current_EPS = each_eps
        if self.debug == True:
            print("Current Eps: {}".format(each_eps))
            print("Current Hour: {}".format(self.current_hour))
            print("Current Stock: {}".format(self.bike_stock[self.current_hour]))
            print("Bikes Moved in Last Hour: {}".format(self.bike_moved))
            print("Collect {} rewards".format(self.reward))
            print("Will move {} bikes".format(action))
            print("---")
        #initializ reward to 0
        self.reward = 0
        if action != 0:
            self.update_stock(action)
            self.reward -=0.5*np.abs(action)
        else:
            self.reward -= 0
            
        if self.bike_stock[self.current_hour] > 40:
            self.reward -= 30
            
        if self.bike_stock[self.current_hour] < 5:
            self.reward -= 30
        
        if self.current_hour == 23:
            #if (self.bike_stock[self.current_hour] <= 45)&(self.bike_stock[self.current_hour] > 0):
                #self.reward = 30
                #pass
            #lse: 
                #self.reward = -30
                #pass
            self.done = True
            #self.new_stock = 'terminal'
            self.game_over = True

        # update to next hour
        if self.current_hour != 23:
            self.update_hour()
            self.old_stock = self.bike_stock[self.current_hour - 1]
            self.new_stock = self.bike_stock[self.current_hour]
            #print ('from env line 459 self.exp_bike_stock ',self.exp_bike_stock)
            self.expected_stock = self.exp_bike_stock[self.current_hour - 1]
            if self.current_hour < 23:
                self.expected_stock_new = self.exp_bike_stock[self.current_hour]
            

        return self.current_hour, self.old_stock, self.new_stock, self.expected_stock, self.expected_stock_new, self.reward, self.done, self.game_over
    
    def get_old_stock(self):
        
        return self.old_stock

    def get_expected_stock(self):
        if self.current_hour < 23:
            return self.expected_stock
        else:
            return None
    
    def update_stock(self, num_bike):
        
        # update bike stock based on RL Agent action at t
        if self.current_hour != 23:
            #print('from env update_stock, currenthour+1 is ', self.current_hour+1, ' len(bikestock is) ', len(self.bike_stock))
            for hour in range(self.current_hour+1, len(self.bike_stock)):
                #print ('from env update_stock, hour is ', hour)
                self.bike_stock[hour] += num_bike
                if hour < len(self.bike_stock)-1:
                    self.exp_bike_stock[hour] += num_bike
                
            self.bike_moved = num_bike
        
        else:
            if self.debug == True:
                print("Last Hour. Cannot Move Bikes.")
            pass
        
        return
    
    def update_hour(self):
        
        # update current_hour
        #print('from update_hour current eps ', self.current_EPS)
        updateHour = self.current_hour +1
        #self.current_hour += 1
        self.current_hour = updateHour%24
        
        if self.debug == True:
            print("Tick... Forwarded Current Hour ",self.current_hour)
                
        return


    def get_current_hour(self):
        return self.current_hour



    def reset(self,each_eps):
        
        if self.debug == True:
            print("Reset Environment ...")
        
        self.num_hours = 23
        self.current_hour = 0
        #self.bike_stock = self.bike_stock_sim.copy() made by Tian
        self.bike_stock = (self.bike_stock_sim.copy())[each_eps*24:each_eps*24+24]
        #print ('from env reset, bikeStok is ', self.bike_stock,'from env reset, each_eps is ',each_eps)
        #print('from env line 520, self.bike_stock.copy()', self.bike_stock.copy())
        #self.exp_bike_stock = self.exp_bike_stock_sim.copy()
        self.exp_bike_stock = self.bike_stock.copy()
        self.done = False
        self.reward = 0
        self.bike_moved = 0
        self.old_stock = self.bike_stock[0]
        self.new_stock = 0
        self.expected_stock = self.exp_bike_stock[0]
        self.expected_stock_new = 0
        #return (self.current_hour, self.old_stock, self.new_stock)
        
    def current_stock(self):
       #print('from env current_stock eps is ', self.current_EPS)
        pointer = (self.current_EPS)*24+self.current_hour
       #print ('from env current_stock pointer is: ', pointer)
        
        #return self.bike_stock[self.current_hour]
        return self.bike_stock[pointer]
    
    def get_sim_stock(self):
        
        return self.bike_stock 
   
