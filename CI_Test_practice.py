import numpy as np 
import pandas as pd 
import sys
from confidence_intervals_TEST_COPY import Coverage
import matplotlib as mpl 
import matplotlib.pyplot as plt 
plt.switch_backend("TkAgg")
import seaborn as sns 

s = "hello"

print(s[1:2])

# count_dict = {    "percentile":		    0,
# 				   "normal":			0,
# 				   "basic":				0,
# 				   "bias_corrected":	0,
# 				   "BC_a":				0}


# iters = 1000

# for i in range(iters):
# 	#Input
# 	rho_goal = np.random.normal(size = 1)

# 	#Creation of inital rho_hat
# 	rho_hats = []
# 	data = np.random.normal(rho_goal, size = 1000)
# 	rho_hat = np.mean(data)

# 	#Creation of bootstrapped rho_hats
# 	for j in range(1000):
	
# 		data = np.random.normal(rho_hat,size = 1000)
# 		rho_sub_i = np.mean(data)
# 		rho_hats.append(rho_sub_i)


# 	coverage = Coverage(rho_hats, rho_hat, confidence = 0.68)

# 	coverage.coverage_orch()

# 	result = coverage.return_intervals()

# 	for key in result.keys():
# 		if result[key][0] <= rho_goal and result[key][1] >= rho_goal:
# 			count_dict[key] += 1

# 	if i%100 == 0:
# 		print("Progress update: %s / %s" %(i,iters))

# print(count_dict)


