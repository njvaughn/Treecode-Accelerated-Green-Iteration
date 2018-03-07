'''
Created on Jan 23, 2018

@author: nathanvaughn
'''

import cProfile
import pstats
import socket
import sys

import TreeStruct
# cProfile.run('re.compile("foo|bar")')
cProfile.run('TreeStruct.TestTreeForProfiling()','profilestats')
p = pstats.Stats('profilestats')
p.strip_dirs().sort_stats('time').print_stats(20)