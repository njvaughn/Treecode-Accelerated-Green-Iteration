'''
Created on Jan 23, 2018

@author: nathanvaughn
'''

import cProfile
import pstats
import socket
import sys
# sys.path.append('methods/')

import Tree
# cProfile.run('re.compile("foo|bar")')
cProfile.run('Tree.TestTreeForProfiling()','profilestats')
p = pstats.Stats('profilestats')
p.strip_dirs().sort_stats('time').print_stats(20)