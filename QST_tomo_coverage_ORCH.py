###########################################################################
#
# Quantum State Tomography Simulation Orchestrator
# **Conduct all tests here**
#
# Author: Sam Showalter
# Date: June 5, 2018
#
###########################################################################

###########################################################################
# Module and library imports 
###########################################################################

#Logistic and system-based imports
import os
import sys
import smtplib

#Data analysis and scientific computing libraries
import numpy as np 
from numpy import linalg as LA
from scipy.linalg import sqrtm

# Compatible with both Anaconda and regular Python distributions
import cvxopt
import ncpol2sdpa

#Package specific imports
from logger import Log
from QST_sim import QSTSim
from rho_estimation import RhoEstimator 
from measurement_sim import MeasSim 
from confidence_intervals import Coverage
from linear_inversion import LinearInversion
from diluted_mle import DilutedMLE 

###########################################################################
# -- Email Script -- #
###########################################################################

# Send emails about progress
def sendProgressEmail(subject, message):
	#Establish server
	server = smtplib.SMTP('smtp.nist.gov', 25)

	#Send the final message
	final_message = 'Subject: {}\n\n{}'.format(subject, message)
	server.sendmail("samuel.showalter@NIST.gov", "samuel.showalter@NIST.gov", final_message)

	#Quit the server
	server.quit()

#sendProgressEmail("Test", "This is just a test.")

###########################################################################
# -- Reference Masterlog -- #
###########################################################################

MasterLog = Log("Hi-There-Test","Not Applicable","Not Applicable")

###########################################################################
# -- Code -- Testing -- #
###########################################################################


print("######################################################################################################################################################")
###########################################################################

# SET SEED FOR EXECUTION
np.random.seed(22)

# 1.) How often to density matrices need to be corrected?

# USE POLY-Y STATE AS THE PURE STATE HERE

MasterLog1 = Log("Main-Diag-1e10-Stop-Crit-Log","Not Applicable", "Not Applicable")

#Noise percentage for pure states
noise = [0.0, 0.01, 0.05]
#Used for both the initial and bootstrapped trials
trials = [1000,10000]

count = 0

# Define theta and phi for the blocsphere
theta = np.pi / 4
phi = np.pi / 4

#Make density matrix
main_diag = 0.5*np.matrix([[(1 + np.cos(theta)), (np.cos(phi)*np.sin(theta) - np.sin(phi)*np.sin(theta)*1j)],
			   [((np.cos(phi)*np.sin(theta) + np.sin(phi)*np.sin(theta)*1j)), (1 - np.cos(theta))]])


for i in range(10):
#Iterate through all options
		for noise_level in noise:
			for trial_num in trials:
				count += 1
			

				#try:
				#Correction for Negative Matrices
				test1 = QSTSim(	MeasSim(#Rho_goal
										 main_diag,
										#Noise_perc
										noise_level,
										#initial rho hat trials
										trial_num,
										#Rho sub_experiments
										100,
										#sub_experiment trials
										trial_num),

								#Solver(s)
								[ DilutedMLE(), LinearInversion()],

								#Log will have 40 individual tests
								log = MasterLog1,

								#Coverage defaults to none
								coverage = Coverage(confidence = 0.68))


				if count % 100 == 0:
					print("\n\nProgress Update: %s\n\n"%str(count))
					# sendProgressEmail("Test 1 Progress", "The tomography simulation from test 1 with %s noise"%(str(noise_level)) +
					# 				  " and %s trials using the %s solver method completed successfully in %s seconds."%(trial_num, 
					# 				  																					test1.RhoEst.Solver.solver_method,
					# 				  																					test1.duration_sec))

				#except Exception as e:
					#print(str(e))
					#sendProgressEmail("Test 1 Progress", "The tomography simulation from test 1 Failed:\n\n%s\n"%(str(e)))



MasterLog1.saveMasterLog()

# data = np.matrix([[0.53300125+0.j,         0.10022995-0.48873804j],

#  [0.10022995+0.48873804j, 0.46699875+0.j        ]])

#print(data)
# theta = np.pi / 4
# phi = np.pi / 4
# #print(np.matmul(data,data).trace())
# main_diag = 0.5*np.matrix([[(1 + np.cos(theta)), (np.cos(phi)*np.sin(theta) - np.sin(phi)*np.sin(theta)*1j)],
# 			   [((np.cos(phi)*np.sin(theta) + np.sin(phi)*np.sin(theta)*1j)), (1 - np.cos(theta))]])

# print(main_diag)


# print(np.outer(np.array([1,0]),
# 			   np.array([1,0])))
print("######################################################################################################################################################")
###########################################################################


# # 2.) How do fidelities change with number of trials, noise, and by solver

# MasterLog2 = Log("Fid-Master-Log","Not Applicable", "Not Applicable")

# noise = [0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]   
# solvers = [LinearInversion(), DilutedMLE()]
# trials = [10,100,1000,5000,10000,50000,100000,500000,1000000]


# for noise_level in noise:
# 	for solver in solvers:
# 		for trial_num in trials:

# 			try:
# 				#Test
# 				test1 = QSTSim(	MeasSim(#Rho_goal
# 										np.matrix([[0.5,-0.5j],
# 				[ 0.5j,0.5]])
# 										#Noise_perc
# 										noise,
# 										#initial rho hat trials
# 										trials,
# 										#Rho sub_experiments
# 										10,
# 										#sub_experiment trials
# 										10),

# 								#Solver
# 								solver,

# 								#Coverage 
# 								coverage = Coverage(method = "all"),

# 								log = MasterLog)

# 				sendProgressEmail("Test 2 Progress", "The tomography simulation from test 1 with %s noise"%(str(noise_level)) +
# 									  " and %s trials using the %s solver method completed successfully in %s seconds."%(trial_num, 
# 									  																					test1.RhoEst.Solver.solver_method,
# 									  																					test1.duration_sec))

# 			except Exception as e:
# 				sendProgressEmail("Test 2 Progress", "The tomography simulation from test 1 Failed:\n\n%s\n"%(str(e)))

# MasterLog2.saveMasterLog()

print("######################################################################################################################################################")
###########################################################################




###########################################################################
# Weird non-diagonal re-composition example
###########################################################################

# t = np.matrix([[0.5,-0.5j],
# 				[ 0.5j,0.5]])

# print("Input matrix:")
# print(t)
# print("")

# eigval, eigvec = LA.eigh(t)

# print("eigenvalues: \n%s\n"%eigval)
# print("eigenvectors: \n%s\n"%eigvec)


# print("Eigenector outer product example with first eigenvector: \n%s\n"%np.outer(eigvec[0], eigvec[0].getH()))

# s = 0

# for eigv in range(len(eigvec)):
# 	s += max(eigval[eigv],0)*(np.outer(eigvec[:,eigv],eigvec[:,eigv].getH()))

# print("Matrix recomposition: \n%s"%s)


# ms = np.matrix(#Computational basis
# 											[np.array([1, 0, 0, 0]),
# 											 np.array([0, 0, 0, 1]),

# 											 #Right left basis
# 											 0.5*np.array([1, 1, 1, 1]),
# 											 0.5*np.array([1, -1, -1, 1]),

# 											 #Plus- minus basis
# 											 0.5*np.array([1, 0+1j, 0-1j, 1]),
# 											 0.5*np.array([1, 0-1j, 0+1j, 1])]
# 											)

# for i in range(len(ms)):
# 	print(ms[i].reshape((2,2), order = "C"))
# 	print(ms[i].reshape((1,4), order = "C"))