###########################################################################
#
# Rho Estimation Engine
# Author: Sam Showalter
# Date: June 5, 2018
#
###########################################################################

###########################################################################
# Module and library imports (all dependencies needed to run this project)
###########################################################################

#Logistic and system-based imports
import sys
import datetime as dt 
import copy

#Data analysis and scientific computing libraries
import numpy as np 
from numpy import linalg as LA
from scipy.linalg import sqrtm

#Package specific imports
from logger import Log
from linear_inversion import LinearInversion
from diluted_mle import DilutedMLE

###########################################################################
# Class and constructor
###########################################################################

class RhoEstimator():
	"""
	Estimates Rho after being given data from the measurement simulator, implements this with different
	solver methods as well as for bootstrapping and initial discovery of rho_hat

	Attributes:

			### Information pertaining to class object interation ###

		MeasSim:				Measurement simulation object with all measurement data
		Solver:					Solver engine to be used to estimate rho_hat

			### Information pertaining to the outer loop analysis ###

		rho_hat:				Rho_hat value found from initial data collected using rho_true
		rho_hat_fid:			Fidelity of the rho_hat initially found

			### Information pertaining to the innter loop bootstrapping analysis ###

		boot_rhos:				Rho_sub_i_hats found from bootstrapped sample data and probabilities
		boot_fids:				Fidelity values relative to rho_goal for each boot_rho derived

	"""


	def __init__(self, 	measurement_sim, 
						solver = LinearInversion()):
		"""
		Instantiates RhoEstimator class.

		Attributes:

			measurement_sim:	Measurement simulator class object with measurement data
			solver:				kwarg that articulates the solver to be used in further
								analysis. Defaulted to Linear Inversion

		"""
		
		# Instantiate information from measurement simulation
		self.MeasSim = measurement_sim 

		# Establish variables necessary to store rho_hat information
		self.rho_hat = None 
		self.rho_hat_fid = None 
		self.rho_hat_purity = None

		# Establish variables necessary to collect bootstrap information
		self.boot_rhos = []
		self.boot_fids = []
		self.boot_purities = []

		#Instantiate solver
		self.Solver = solver

		#Instantiate log from orchestrator (Master log not used)
		self.BootLog = Log("Not Applicable", "s%s_%s_n%s_t%s_Boot-Log"%(str(self.MeasSim.seed).replace("-",""), 
																		self.Solver.solver_method, 
																		self.MeasSim.noise_perc, 
																		self.MeasSim.rho_hat_trials), 
																		"Verbose-Log")

		#Set boot log filename
		self.BootLogFilename = self.BootLog.boot_log_name + ".csv"

		#Determine rho_true fidelity
		self.MeasSim.rho_true_fidelity = self.calculateFidelity(self.MeasSim.rho_true)

		#Set the estimator for the Solver
		self.Solver.setEstimator(self)

	###########################################################################
	# Orchestrator
	###########################################################################

	# General function that solves for Rho dynamically depending on solver
	# Also generates fidelity measurements after
	def estimateRho(self, bootstrap = False):
		"""
		Dynamic function to solve for rho. Will dynamically implement different
		solver functions and also vary its execution based on whether or not we 
		are bootstrapping. Also calculated fidelity, another dynamic function
		that depends on the bootstrapping parameter.
		
		This estimation function also dynamically determines whether or not bootstrapping was requested at all

		Args:
			bootstrap:		Whether or not we are bootstrapping
		"""
		self.Solver.solveForRho(bootstrap)

		#Initialize the mean bootstrapped metrics for logging
		self.mean_boot_log_lik = "-"
		self.mean_boot_purity = "-"

		#Inform measurement simulator error correcting
		#That rho_hat has been created
		if (not bootstrap):
			self.MeasSim.rho_hat = self.rho_hat

		#If we just finished bootstrapping
		else:
			
			#If bootstrapping was requested at all
			if self.MeasSim.rho_sub_experiments > 0:

				#Summary data to be stored for bootstrapping (log_lik and purity)
				self.mean_boot_log_lik = float(self.BootLog.boot_log["Boot_Rho_Log_Lik"].mean())
				self.mean_boot_purity = float(self.BootLog.boot_log["Boot_Rho_Purity"].mean())

				#Save the bootstrap log
				self.BootLog.saveBootLog()

			#If bootstrapping was not requested (no sub-experiments)
			else:
				self.BootLogFilename = "-"

	###########################################################################
	# Public helper functions for different rho calculations
	###########################################################################

	def calculateFidelity(self, Rho):
		"""
		Public function that calculates the fidelity of matrices based on whether or not
		the user wants to use bootstrapping or analyze the initial rho_hat found via rho_true.
		Adds fidelities to their appropriate Tomography variable for later use.

		Args:
			Rho:			Density matrix from boot_rhos or initial rho_hat

		Returns:
			fidelity:		Fidelity calculation of a matrix relative to rho_goal
		"""

		# If the function is called for the initial rho calculations.
		fidelity = self.__fidelityHelper(Rho).real
		return fidelity

	def calculateLogLikelihood(self, rho, probs, trials):
		"""
		Calculate the log-likelihood of the estimated rho and return the information. You can
		do this by summing the natural log of the trace of POVM measurement E and rho, multiplied by the
		frequency of each occurrence.

		Args:
			rho:				rho_hat to be given a likelihood
			probs:				Empirically determined probability array
			trials:				Number of measurements in the experiment

		Returns:
			log_likelihood:		Log-likelihood scalar
		"""

		#Initialize number of trials
		N = trials

		#Initialize log_likelihood variable
		log_likelihood = 0

		#Iterate through all the measurement bases
		for op_index in range(0, self.MeasSim.meas_ops.shape[0], 2):

			#Re-shape pi_zero and pi_one
			pi_zero = self.MeasSim.meas_ops[op_index].reshape((self.MeasSim.dims,self.MeasSim.dims), order = "F")
			pi_one = self.MeasSim.meas_ops[op_index+1].reshape((self.MeasSim.dims,self.MeasSim.dims), order = "F")

			#Find pi_zero and pi_one frequencies for the experiment
			pi_zero_freq = float((probs[op_index] + (1 - probs[op_index + 1]))*N)
			pi_one_freq = float(((1 - probs[op_index]) + (probs[op_index + 1]))*N)

			#Initialize trace_pi_zero and trace_pi_one
			ln_trace_pi_zero = 0
			ln_trace_pi_one  = 0
			
			#Get natural log of the trace for pi zero and pi one
			ln_trace_pi_zero = np.log(float((np.matmul(rho, pi_zero)).trace().real))
			ln_trace_pi_one = np.log(float((np.matmul(rho, pi_one)).trace().real))

			#Add pi_zero component to the sum
			log_likelihood += pi_zero_freq * ln_trace_pi_zero

			#Add pi_one component to the sum
			log_likelihood += pi_one_freq * ln_trace_pi_one

		#Return linear combination of R and the identity matrix (via epsilon)
		return log_likelihood/N

	###########################################################################
	# Private helper functions, ordered from low granularity to high
	###########################################################################
	def __fidelityHelper(self, matrix):
		"""
		Private helper method for calculate Fidelity. Actually does fidelity calculation
		for quantum state tomography. All matrices compared to rho_goal

		Args:
			matrix:			Density matrix from boot_rhos or initial rho_hat

		Returns:
			fidelity:		Float from [0,1] representing the inter-matrix fidelity

		Raises:
			ValueError:		Double checks to make sure that the fidelity is not imaginary
		"""

		#Calculate fidelity of matrix relative to rho_goal
		fidelity = ((sqrtm(np.matmul(np.matmul(sqrtm(self.MeasSim.rho_goal), 
												 	 matrix),
								  	       	   sqrtm(self.MeasSim.rho_goal)))).trace())**2

		#Make sure that imaginary component is arbitrarily small (python float imperfections)
		if (fidelity.imag - 0) > self.MeasSim.tol:
			print("fidelity BAD \n%s\n"%fidelity)
			raise ValueError("Imaginary fidelity discovered. Check calculations.")

		#If there are no problems, return the fidelity
		else:
			return fidelity

	
