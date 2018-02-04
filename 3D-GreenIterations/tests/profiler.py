'''
Created on Jan 23, 2018

@author: nathanvaughn
'''

import cProfile
import pstats
import socket
import sys
sys.path.append('methods/')

import GreenIterations
# cProfile.run('re.compile("foo|bar")')
cProfile.run('GreenIterations.run(socket.gethostname(),int(sys.argv[1]),int(sys.argv[2]),float(sys.argv[3]))','profilestats')
p = pstats.Stats('profilestats')
p.strip_dirs().sort_stats('time').print_stats(20)