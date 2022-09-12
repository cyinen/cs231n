'''
Author: Yin Chen
Date: 2022-09-11 02:56:32
LastEditors: Yin Chen
LastEditTime: 2022-09-11 03:18:29
Description: 
'''
import os

class logger(object):
    def __init__(self, log_path="log.txt"):
        self.log_path = log_path
    def write(self, msg):
        print(msg)
        if not msg.endswith("\n"):
            msg += "\n"
        with open(self.log_path, "a") as f:
            f.write(msg)
            f.close()
        