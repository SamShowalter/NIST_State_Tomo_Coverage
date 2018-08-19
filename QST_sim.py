###########################################################################
#
# Quantum State Tomography Simulation Test Object (Shell)
# Author: Sam Showalter
# Date: June 5, 2018
#
###########################################################################


###########################################################################
# Module and library imports 
###########################################################################

#Logistic and system-based imports
import sys
import copy
import datetime as dt 

#Data analysis and scientific computing libraries
import numpy as np 
from numpy import linalg as LA
from scipy.linalg import sqrtm

#Package specific imports
from rho_estimation import RhoEstimator 
from logger import Log
from measurement_sim import MeasSim 
from confidence_intervals import Coverage


###########################################################################
# Class and Constructor
###########################################################################

class QSTSim():

	"""
	Performs quantum state tomography simulation. Connects all facets of tomography
	simulation, estimation, and coverage into a single test

	Attributes:

					### Information pertaining to class object interation ###

		MeasSim:		Measurement simulation class object. Contains measurement data
		Solver:			Chosen solver to determine rho_hat. Creates RhoEstimatorClass
						along with the relevant solver class. 
		Coverage:		Coverage class for constructing bootstrapped confidence intervals
		MasterLog:		Master Log to be used for execution data storage

	"""

	def __init__(self,
				 meas_sim,
				 solver,
				 log,
				 coverage = None):
		"""
		Instantiates the QST Simulation class (test objects). Runs the entire simulation from the
		constructor.

		Args:
			meas_sim:				Measurement simulation class object. Contains measurement data
			solver:					Chosen solver to determine rho_hat. Creates RhoEstimatorClass
									along with the relevant solver class. 
			log:					Master Log to be used for execution data storage
			coverage:				kwarg Coverage class for constructing bootstrapped confidence intervals
									If set to None, coverage package will not be run

		Raises:
			ValueError:				If coverage is selected to be run but there are no bootstrapped
									samples to use, then raise an error and end the execution

		"""

		#Start date and time metadata of execution
		self.execution_date_start = dt.datetime.now().date()
		self.execution_time_start = dt.datetime.now().time().strftime("%H.%M.%S")
		utc_start_time = dt.datetime.utcnow()

		#Create measurement simulation class
		self.MeasSim = meas_sim

		#Set master log
		self.MasterLog = log

		#Run the execution to get measurements for first Rho_hat
		self.MeasSim.measSim()

		# #Debugging
		# print("\nInitial rho Frequencies.")
		# print(self.MeasSim.rho_hat_probs)

		#Initialize solver iteration count to False 
		#(this allows the same data to be used for both solvers)
		solver_iteration_count = False

		#Determine how many solvers to use (will be list if more than one)
		if type(solver) != list:
			solver = [solver]

		#Iterate through all the solver methods requested
		for solver_method in solver:

			#Instantiate a Rho estimator instance
			self.RhoEst = RhoEstimator(self.MeasSim, solver_method)

			#Set master log for bootstrapped Rho
			self.RhoEst.BootLog.master_log_name = self.MasterLog.master_log_name

			#Use RhoEst to solve for Rho_hat
			self.RhoEst.estimateRho()

			# #Debugging
			# print("\nRho_hat_initial:")
			# print(self.RhoEst.rho_hat)

			# #Debugging
			# print("\nFidelity")
			# print(self.RhoEst.rho_hat_fid)

			#Return to MeasSim to conduct bootstrapping
			if not solver_iteration_count:
				self.MeasSim.measSim(bootstrap = True)
				solver_iteration_count = True

			#Back again to RhoEst to estimate Rho for bootstrap samples
			self.RhoEst.estimateRho(bootstrap = True)

			# #Debugging (only print if you are bootstrapping)
			# print("\n\nFirst sample from rho_hats")
			# print(self.RhoEst.boot_rhos[0])

			# #Debugging
			# print("\nRho Est log")
			# print(self.RhoEst.BootLog.boot_log)

			#####################################################################
			# Start of bootstrapping confidence interval code
			#####################################################################

			#If coverage is supposed to be run
			if coverage is not None:

				#Check to make sure there are bootstrapped experiments requested
				if self.MeasSim.rho_sub_experiments == 0:
					raise ValueError("Coverage may not be completed without any" +
									" bootstrapped samples. Check bootstrap inputs.")

				#Assign coverage object
				self.Coverage = coverage

				#Set the estimator data for the coverage object
				self.Coverage.RhoEst = self.RhoEst 

				#Run coverage orchestrator
				self.Coverage.coverage_orch()

			#Default coverage if it is not run
			else:
				#This code exists to ensure the logger includes N/A ("-")
				#values when it stores records. This way, we know if the
				#Coverage package was supposed to be run.
				self.Coverage = Coverage() 
				self.Coverage.RhoEst = self.RhoEst 

			#####################################################################
			# Add Test metadata to log 
			#####################################################################

			#Duration metadata for execution
			self.duration_sec = (dt.datetime.utcnow() - utc_start_time).total_seconds()

			#Add execution data to the master log
			self.MasterLog.addMasterLogRecord(self)


	


	


	
					