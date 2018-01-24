'''
Created on Jan 23, 2018

@author: nathanvaughn
'''

import cProfile
import pstats

import main
# cProfile.run('re.compile("foo|bar")')
cProfile.run('main.run()','profilestats')
p = pstats.Stats('profilestats')
p.strip_dirs().sort_stats('time').print_stats(20)