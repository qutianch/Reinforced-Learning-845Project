#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 15:54:01 2018

@author: Ian, Prince, Brenton, Alex

This is the main workflow of the RL program.

"""

from training import trainer
import helper

if __name__ == "__main__":

    # Get Initial Parameters    
    episode_list, data, ID, brain, algorithm, model_based, station_history = helper.user_input()


    # Set Up a Training Environment

    if brain != 'exit':
        trainer = trainer(station_history)
        trainer.start(episode_list, data, logging=True, env_debug=False, rl_debug=False, brain=brain, ID=ID,
                      model_based=model_based, algorithm=algorithm)
    else:

        print ('User choice,  program exiting')