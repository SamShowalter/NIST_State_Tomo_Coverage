###########################################################################
#
# Quantum State Tomography Result Visualization
# **Conduct all visualization tests here**
#
# Author: Sam Showalter
# Date: June 5, 2018
#
###########################################################################

###########################################################################
# Module and library imports 
###########################################################################

#System-based imports
import os
import sys
import datetime as dt 
import glob

# Visualization imports
import matplotlib as mpl 
import matplotlib.pyplot as plt 
plt.switch_backend("TkAgg")
import seaborn as sns

# Data analysis imports
import numpy as np 
import pandas as pd 

def isValid( s):
        """
        :type s: str
        :rtype: bool
        """
        stack = []
        ref_dict = {")":"(", "}":"{", "]": "["}
        
        for item in range(len(s)):
            if s[item] in ref_dict.keys():
                if len(stack) == 0 or ref_dict[s[item]] != stack[-1]:
                    return False
                else:
                    del stack[-1]
            else:
                stack.append(s[item])
        
        return len(stack) == 0

print(isValid("[[[[[[[{}]]]]]][]]"))
###########################################################################
# Data Visualizations --ADD YOUR INPUTS HERE!
###########################################################################

#WHICH SOLVER IS BEING USED FOR THESE CHARTS
SOLVER = "Diluted_MLE"

#NOISE PERCENTAGE TO USE
noise = 0.01

#NUMBER OF TRIALS
trials = 1000

# DATA TO BE USED
# CHANGE THESE TO YOUR OWN LOCAL DIRECTORY

# Non-pure bulk data
# data = pd.read_csv("C:\\Users\\srs9\\Documents\\State_Tomo_Coverage_Data\\Analysis_Data\\Bulk-Data-Log_2018-06-27_10.30.50.csv")

# #Pure bulk data -- USED FOR BULK DATA LOG
#data = pd.read_csv("C:\\Users\\srs9\\Documents\\State_Tomo_Coverage_Data\\Main-Diagonal-Large-Data-Log_2018-07-09_09.35.41.csv")

#print(data[(data["Noise_Perc"] == 0.0) & (data["Solver_Method"] == "Linear_Inversion")].loc[:,"Rho_Hat_Purity"].mean())

# Positivity Correction Data -- USED FOR POS CORR LOG
# data = pd.read_csv("C:\\Users\\srs9\\Documents\\State_Tomo_Coverage_Data\\Analysis_data\\Pos-Corr-Log_2018-06-27_10.28.18.csv")

###########################################################################################################
# Visualization of number of positivity corrections -- ONLY UNCOMMENT WHEN POSITIVITY CORRECTION DATA USED
###########################################################################################################

#finaldata = pd.read_csv("C:\\Users\\srs9\\Documents\\State_Tomo_Coverage_Data\\Pos_Corr_Data\\Main_Diagonal\\Main-Diag-Pos_Corr_Data.csv")
# data = pd.read_csv("C:\\Users\\srs9\\Documents\\State_Tomo_Coverage_Data\\Pos_Corr_Data\\Main_Diagonal\\Main-Diagonal-Pos-Corr-Data-Log_2018-07-11_15.47.29.csv")


# noise = [0.0, 0.01, 0.05, 0.2]
# trials = [10, 100,1000,5000,10000,50000,75000,100000]
# res = []

# for noise_p in noise:
# 	for t in trials:
# 		datanew = data[(data["Noise_Perc"] == noise_p) & (data["Initial_Rho_Hat_Trials"] == t)]

# 		prob = datanew["Rho_Hat_Pos_Corr"].sum() / float(len(datanew))

# 		uncert = np.sqrt(prob*(1-prob) / len(datanew))

# 		res.append((noise_p, t,prob,uncert))


# for i in res:
# 	print(" ".join([str(j) for j in i]))


# sns.factorplot(x="Trials", y="Pos_Corr_Perc", hue="Noise", data=data,
#                    capsize=.1, size=10, aspect=.75)
# plt.suptitle('Main_Diagonal Factor Plot of Pos_Corr_Perc by Noise Percentage and Trial #', fontsize = 20)

# plt.show()

###########################################################################################################
# Visualization of number of positivity corrections -- ONLY UNCOMMENT WHEN POSITIVITY CORRECTION DATA USED
###########################################################################################################


# #Make Pos_Corr_Percent column
# data["Pos_Corr_Perc"] = data["Boot_Rho_Hat_Pos_Corr"] / data["Rho_Sub_Experiments"]

# #Refine the data for positivity corrections
# data_trials = data[(data["Noise_Perc"] == noise) & (data["Solver_Method"] == SOLVER)]
# data_noise= data[(data["Initial_Rho_Hat_Trials"] == trials) & (data["Solver_Method"] == SOLVER)]

# #Box plots by Noise Percent and Initial Rho_Hat_Trials
# # sns.boxplot(y = "Pos_Corr_Perc", x = "Noise_Perc", data = data_noise)
# # plt.show() 

# sns.boxplot(y = "Pos_Corr_Perc", x = "Initial_Rho_Hat_Trials", data = data_trials)
# plt.suptitle('Box Plot of Rho_Hat Positivity Correction by Trial #, Noise: %s, Solver: %s'%(str(noise), str(SOLVER)), fontsize = 20)
# #Show the plots
# plt.show()

#############################################################################################
# Capture information for Rho_True for confidence interval analysis
#############################################################################################

# #read in data
# data = pd.read_csv("C:\\Users\\srs9\\Documents\\State_Tomo_Coverage_Data\\Pauli-Y-LARGE-Interval-Data-Log_2018-07-18_09.37.57.csv")

# fid_pure = ["Fid", "Purity"]

# #get columns ready for fidelity
# data["Normal_CI_" + fid_pure[0] + "_Success"] = ((data["CI_" + fid_pure[0] + "_Norm_Low"] <= data["Rho_True_Fidelity"]) & (data["CI_" + fid_pure[0] + "_Norm_High"] >= data["Rho_True_Fidelity"]))
# data["BC_CI_" + fid_pure[0] + "_Success"] =((data["CI_" + fid_pure[0] + "_BC_Low"] <= data["Rho_True_Fidelity"]) & (data["CI_" + fid_pure[0] + "_BC_High"] >= data["Rho_True_Fidelity"]))
# data["BC_a_CI_" + fid_pure[0] + "_Success"] = ((data["CI_" + fid_pure[0] + "_BC_a_Low"] <= data["Rho_True_Fidelity"]) & (data["CI_" + fid_pure[0] + "_BC_a_High"] >= data["Rho_True_Fidelity"]))
# data["Percentile_CI_" + fid_pure[0] + "_Success"] = ((data["CI_" + fid_pure[0] + "_Percentile_Low"] <= data["Rho_True_Fidelity"]) & (data["CI_" + fid_pure[0] + "_Percentile_High"] >= data["Rho_True_Fidelity"]))
# data["Basic_CI_" + fid_pure[0] + "_Success"] = ((data["CI_" + fid_pure[0] + "_Basic_Low"] <= data["Rho_True_Fidelity"]) & (data["CI_" + fid_pure[0] + "_Basic_High"] >= data["Rho_True_Fidelity"]))

# #get columns ready for fidelity
# data["Normal_CI_" + fid_pure[1] + "_Success"] = ((data["CI_" + fid_pure[1] + "_Norm_Low"] <= data["Rho_True_Purity"]) & (data["CI_" + fid_pure[1] + "_Norm_High"] >= data["Rho_True_Purity"]))
# data["BC_CI_" + fid_pure[1] + "_Success"] =((data["CI_" + fid_pure[1] + "_BC_Low"] <= data["Rho_True_Purity"]) & (data["CI_" + fid_pure[1] + "_BC_High"] >= data["Rho_True_Purity"]))
# data["BC_a_CI_" + fid_pure[1] + "_Success"] = ((data["CI_" + fid_pure[1] + "_BC_a_Low"] <= data["Rho_True_Purity"]) & (data["CI_" + fid_pure[1] + "_BC_a_High"] >= data["Rho_True_Purity"]))
# data["Percentile_CI_" + fid_pure[1] + "_Success"] = ((data["CI_" + fid_pure[1] + "_Percentile_Low"] <= data["Rho_True_Purity"]) & (data["CI_" + fid_pure[1] + "_Percentile_High"] >= data["Rho_True_Purity"]))
# data["Basic_CI_" + fid_pure[1] + "_Success"] = ((data["CI_" + fid_pure[1] + "_Basic_Low"] <= data["Rho_True_Purity"]) & (data["CI_" + fid_pure[1] + "_Basic_High"] >= data["Rho_True_Purity"]))


# #Visualize the confidence intervals
# noise = [0.0, 0.01, 0.05]
# solvers = ["Linear_Inversion", "Diluted_MLE"]
# trials = [1000,10000]

# res = []
# for f in fid_pure:
# 	for s in solvers:
# 		for n in noise:
# 			for t in trials:
# 				tdata = data[(data["Noise_Perc"] == n) & (data["Initial_Rho_Hat_Trials"] == t) & (data["Solver_Method"] == s)]

# 				res.append((f, s, n, t, tdata["Normal_CI_" + f + "_Success"].mean(), 
# 								 tdata["BC_CI_" + f + "_Success"].mean(),
# 								 tdata["BC_a_CI_" + f + "_Success"].mean(),
# 								 tdata["Percentile_CI_" + f + "_Success"].mean(),
# 								 tdata["Basic_CI_" + f + "_Success"].mean()))

# for i in res:
# 	print(" ".join([str(j) for j in i]))

#############################################################################################
# Comparison of Lower and Upper Bound confidence intervals for Basic Method
#############################################################################################
# dic = {0:"Fidelity", 1:"Purity"}
# p_type = [1]

# for p in p_type:
# 	for s in solvers:
# 		for n in noise:
# 			for t in trials:
# 				tdata = data[(data['Noise_Perc'] == n) & (data["Initial_Rho_Hat_Trials"] == t) & (data["Solver_Method"] == s)]

# 				d = tdata[["Rho_True_Purity", "Rho_Hat_Purity", "CI_" + fid_pure[p] + "_BC_Low","CI_" + fid_pure[p] + "_BC_High" ]]

# 				sns.boxplot(x = "variable", y = "value", data = pd.melt(d))
#  				plt.suptitle('Box Plot of %s BC Method CI by Noise Perc and Trial #; Solver: %s, Noise: %s, Trials: %s'%(dic[p], s, n,t), fontsize = 20)
#  				plt.show()


#############################################################################################
# Interval length Information for Rho_True for confidence interval analysis
#############################################################################################


# p_type = [0,1]

# #get columns ready for Fidelity
# data["Normal_CI_" + fid_pure[0] + "_Length"] = (data["CI_" + fid_pure[0] + "_Norm_High"] - data["CI_" + fid_pure[0] + "_Norm_Low"])
# data["BC_CI_" + fid_pure[0] + "_Length"] =(data["CI_" + fid_pure[0] + "_BC_High"] - data["CI_" + fid_pure[0] + "_BC_Low"])
# data["BC_a_CI_" + fid_pure[0] + "_Length"] = (data["CI_" + fid_pure[0] + "_BC_a_High"] -data["CI_" + fid_pure[0] + "_BC_a_Low"])
# data["Percentile_CI_" + fid_pure[0] + "_Length"] = (data["CI_" + fid_pure[0] + "_Percentile_High"] - data["CI_" + fid_pure[0] + "_Percentile_Low"])
# data["Basic_CI_" + fid_pure[0] + "_Length"] = (data["CI_" + fid_pure[0] + "_Basic_High"] -data["CI_" + fid_pure[0] + "_Basic_Low"])

# #get columns ready for Purity
# data["Normal_CI_" + fid_pure[1] + "_Length"] = (data["CI_" + fid_pure[1] + "_Norm_High"] - data["CI_" + fid_pure[1] + "_Norm_Low"])
# data["BC_CI_" + fid_pure[1] + "_Length"] =(data["CI_" + fid_pure[1] + "_BC_High"] - data["CI_" + fid_pure[1] + "_BC_Low"])
# data["BC_a_CI_" + fid_pure[1] + "_Length"] = (data["CI_" + fid_pure[1] + "_BC_a_High"] -data["CI_" + fid_pure[1] + "_BC_a_Low"])
# data["Percentile_CI_" + fid_pure[1] + "_Length"] = (data["CI_" + fid_pure[1] + "_Percentile_High"] - data["CI_" + fid_pure[1] + "_Percentile_Low"])
# data["Basic_CI_" + fid_pure[1] + "_Length"] = (data["CI_" + fid_pure[1] + "_Basic_High"] -data["CI_" + fid_pure[1] + "_Basic_Low"])

# dic = {0:"Fidelity", 1:"Purity"}

# for p in p_type:
# 	for s in solvers:
# 		for n in noise:
# 			for t in trials:

# 				tdata = data[(data['Noise_Perc'] == n) & (data["Initial_Rho_Hat_Trials"] == t) & (data["Solver_Method"] == s)]

# 				d = tdata[["Normal_CI_" + fid_pure[p] + "_Length",
# 						   "BC_CI_" + fid_pure[p] + "_Length",
# 						   "BC_a_CI_" + fid_pure[p] + "_Length",
# 						   "Percentile_CI_" + fid_pure[p] + "_Length",
# 						   "Basic_CI_" + fid_pure[p] + "_Length"]]

# 				sns.boxplot(x = "variable", y = "value", data = pd.melt(d))
# 				plt.suptitle('Box Plot of %s CI Length by Noise Perc and Trial #; Solver: %s, Noise: %s, Trials: %s'%(dic[p], s, n,t), fontsize = 20)
# 				plt.show()

#############################################################################################
# Pos_Corr_Data visualizations for initial rho_hat only
#############################################################################################
# data = pd.read_csv("C:\\Users\\srs9\\Desktop\\Pos_Corr_Data\\Pos_Corr_Data.csv")
# print(data)
# #Draw a pointplot to show pulse as a function of three categorical factors
# sns.factorplot(x="Trials", y="Pos_Corr_Perc", hue="Noise", data=data,
#                    capsize=.1, size=10, aspect=.75)
# plt.suptitle('Factor Plot of Pos_Corr_Perc by Noise Percentage and Trial #', fontsize = 20)

# plt.show()

#############################################################################################
# Visualization of how trial number effects fidelity
#############################################################################################


# #Fine tune the data
# data_fid1 = data[((data["Noise_Perc"] == noise) & 
# 			(data["Solver_Method"] == SOLVER))]# &
			#(data["Initial_Rho_Hat_Trials"] == trials))]

########################################
# Visualization -- Boxplots for fidelity
########################################

# sns.set(style="ticks")
# f, ax = plt.subplots(figsize=(10, 8))

# # print(data["Initial_Rho_Hat_Trials"].sum())

# sns.boxplot(y = "Rho_Hat_Fidelity", x = "Initial_Rho_Hat_Trials", data = data_fid1)
# plt.suptitle('Box Plot of Rho_Hat Fidelity by Trial #, Noise: %s, Solver: %s'%(str(noise), str(SOLVER)), fontsize = 14)
# plt.show()

# ########################################################
# # Visualization -- factor plot with bounds for fidelity
# ########################################################

# #Fine tune the data
# data_fid2 = data[(data["Noise_Perc"] == noise) & (data["Solver_Method"] != "Rho_True")]
			
# 			#(data["Initial_Rho_Hat_Trials"] == trials))]

# #Style for the plot
# sns.set(style="whitegrid")                                                 
# paper_rc = {'lines.linewidth': 1, 'lines.markersize': 10}                  
# sns.set_context("paper", rc = paper_rc) 

# # print(data["Initial_Rho_Hat_Trials"].sum())

# # Draw a pointplot to show pulse as a function of three categorical factors
# sns.factorplot(x="Initial_Rho_Hat_Trials", y="Rho_Hat_Fidelity", hue="Solver_Method", data=data_fid2,
#                    capsize=.1, size=10, aspect=.75)
# plt.suptitle('Factor Plot of Rho_Hat Fidelity by Solver Method and Trial #, Noise: %s'%str(noise), fontsize = 20)
# plt.show()

# #############################################################################################
# # Likelihood differences between estimated and true states
# #############################################################################################

# # #Fine tune the data
# lik_data = data[(data["Noise_Perc"] == noise)]# & 
# 			#(data["Solver_Method"] == SOLVER)]

# # # ########################################
# # # # Visualization
# # # ########################################

# sns.boxplot(y = "Rho_Hat_Log_Lik", x = "Initial_Rho_Hat_Trials", hue = "Solver_Method", data = lik_data)
# plt.suptitle('Box Plot of Rho_Hat Log Likelihood by Solver Method and Trial #, Noise: %s'%str(noise), fontsize = 14)
# plt.show()


# #############################################################################################
# # Purity differences between estimated and true states
# #############################################################################################

# #Fine tune the data
# purity_data = data[(data["Noise_Perc"] == noise)] #& 

# 			#(data["Solver_Method"] == SOLVER)]
# #plt.ylim(0.9,1)
# sns.factorplot(y = "Rho_Hat_Purity", x = "Initial_Rho_Hat_Trials", hue = "Solver_Method", data = purity_data, size = 10)
# plt.suptitle('Factor Plot of Rho_Hat Purity by Solver Method and Trial #, Noise: %s'%str(noise), fontsize = 20)
# plt.show()

# # ##############################################################################################################
# # #Just look at estimated purities, by noise percent

# #palette={"Linear_Inversion": "b", "Diluted_MLE": "y"}
# purity_data = purity_data[purity_data["Solver_Method"] != "Rho_True"]

# # Draw a nested violinplot and split the violins for easier comparison
# sns.violinplot(x="Initial_Rho_Hat_Trials", y="Rho_Hat_Purity", hue="Solver_Method", data=purity_data, split=True,
#                inner="box", cut = 0, size = 10)

# sns.despine(left=True)
# plt.suptitle('Violin Plot of Rho_Hat Purity by Solver Method and Trial #, Noise: %s'%str(noise), fontsize = 20)
# plt.show()

# #Strip plot of the data
# #############################################################################################################
# #Just look at ALL purities, by noise percent

# # #Just look at estimated purities, by noise percent
# # plt.style.use('ggplot')

# data_pure = data
#  			#(data["Solver_Method"] == SOLVER)]

# # Show each observation with a scatterplot
# ax1 = sns.stripplot(x="Noise_Perc", y="Rho_Hat_Purity", hue="Solver_Method",
#               data=data_pure, dodge=True, jitter=True,
#               alpha=.25, zorder=1,palette={"Linear_Inversion": "b", "Diluted_MLE": "g", "Rho_True": "r"})

# #Show the conditional means
# ax = sns.pointplot(x="Noise_Perc", y="Rho_Hat_Purity", hue="Solver_Method",
#               data=data_pure, dodge=.532, estimator = np.median, join=False, palette={"Linear_Inversion": "w", "Diluted_MLE": "w", "Rho_True": "w"},
#               markers="d", scale=.75, ci=None)

# # check axes and find which is have legend
# # Get the handles and labels. For this example it'll be 2 tuples
# # of length 4 each.
# handles, labels = ax.get_legend_handles_labels()

# # When creating the legend, only use the first two elements
# # to effectively remove the last two.
# l = plt.legend(handles[0:3], labels[0:3], bbox_to_anchor=(0.05, 0.1), loc=2, borderaxespad=0.)


# sns.despine(left=True)
# plt.show()

#############################################################################################
# Likelihood differences between estimated and true states
#############################################################################################