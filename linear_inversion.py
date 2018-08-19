###########################################################################
#
# Linear Inversion Optimizer
# Author: Sam Showalter
# Date: June 5, 2018
#
###########################################################################

###########################################################################
# Module and library imports (all dependencies needed to run this project)
###########################################################################

#Logistic and system-based imports
import sys
import progressbar
import datetime as dt 


#Data analysis and scientific computing libraries
import numpy as np 
from numpy import linalg as LA
from scipy.linalg import sqrtm

###########################################################################
# Class and Constructor
###########################################################################

class LinearInversion():
	"""
	One of potentially many solver classes. Operates within the RhoEstimator
	class with shell functions. No attributes explicitly belong to this class.
	It is exclusively an analysis engine to determine rho_hat.]

	Attributes:
	
				### Attributes pertaining to LinearInversion execution ###

			RhoEst 				RhoEstimator class object containing all necessary
								data for analysis.
			solver_method:		Explicity tells logger what solver method was used

				### Un-used attributes necessary for code dynamism in logging ###

			----- ALL OF THESE APPLY TO MLE EXECUTION ONLY -----
			stopping_crit:			MLE stopping criteria (not used)
			boot_RrhoR_iter:		MLE iterations (always set to none) for bootstrap
			RrhoR_iter:				MLE iterations (always set to none) for initial rho hat
			epsilon_corrections:	MLE logging of number of epsilon adjustments
			verbose_log_filename:	MLE filename for verbose log to examine execution
			iter_bound:				Number of RrhoR iterations before execution times out
			MLE_timeout:			Boolean (1 = True, 0 = False) of MLE timeout occurrence 
									for a sample.
			


	"""

	def __init__(self):
		"""
		Constructor for LinearInversion class object. No parameters are given. 
		Necessary information from the estimator class is provided later in the
		execution by QST_sim.
		"""

		#Defaulting MLE paramters to be "-" to keep everything dynamic (NOT USED)
		self.stopping_crit = "-"
		self.boot_RrhoR_iter = "-"
		self.RrhoR_iters = "-"
		self.epsilon_corrections = "-"
		self.verbose_log_filename = "-"
		self.iter_bound = "-"
		self.MLE_timeout = "-"

		#Set the solver method
		self.solver_method = "Linear_Inversion"

		#Initialize RhoEstimator to None (set later by QSTSim object)
		self.RhoEst = None

	###########################################################################
	# Orchestrator
	###########################################################################

	def solveForRho(self, bootstrap = False):
		"""
		Shell function to provide dynamic compatibility with rho_estimation

		Args:
			bootstrap:			Informs function of bootstrapping

		Generates:
			Rho_hat calculations for initial and bootstrapping samples
		"""
		self.linearInversion(bootstrap)

	def linearInversion(self, bootstrap = False):
		"""
		Derives rho_hat through linear inversion using empirically found data

		First converts the measurement operator matrix into a square matrix
		by muliplying it by its transpose. The equation then becomes

		Rho_hat = (A_trans * A)^(-1) * A_trans * emp_probs

		This also checks to make sure that the resultant matrix is a valid
		density operator and renormalizes the initial matrix calculations.

		Attributes:
			None that are not already encapsulated in other tomography objects

		Pre-conditions:
			initTomoMeasSim has already been run and self.emp_probs is populated

		Generates:
			Initial rho_hat that will be used for later analysis and simulation.

		Raises:
			TypeError:		If bootstrapping measurements have not been taken, bootstrap tomography
							cannot be run.
		"""

		#Reshape meas_ops to be suitable for linear inversion
		meas_ops = self.RhoEst.MeasSim.meas_ops.reshape

		if not bootstrap:
			#Normalizes the final rho_hat operations
			self.RhoEst.rho_hat = self.RhoEst.MeasSim.renormalize(np.matmul(np.matmul(LA.inv((np.matmul(self.RhoEst.MeasSim.meas_ops.T,
													  						   				   		   self.RhoEst.MeasSim.meas_ops))),
									 		   				  				self.RhoEst.MeasSim.meas_ops.T),
									 							  self.RhoEst.MeasSim.rho_hat_probs).reshape(
									 							  (self.RhoEst.MeasSim.dims,self.RhoEst.MeasSim.dims)))
									 							  # Reshape the matrix to dxd

			#Verifies the validity of the Rho_estimation matrices
			self.RhoEst.rho_hat = self.RhoEst.MeasSim.validMatrix(self.RhoEst.rho_hat)

			#Calculate the fidelity of the matrix
			self.RhoEst.rho_hat_fid = self.RhoEst.calculateFidelity(self.RhoEst.rho_hat)

			#Log-likelihood calculation
			self.RhoEst.rho_hat_log_lik = self.RhoEst.MeasSim.calculateLogLikelihood(self.RhoEst.rho_hat, 
																					 self.RhoEst.MeasSim.rho_hat_probs, 
																					 self.RhoEst.MeasSim.rho_hat_trials)

			#Purity calculation
			self.RhoEst.rho_hat_purity = self.RhoEst.MeasSim.calculatePurity(self.RhoEst.rho_hat)

		else:

			#If bootstrapping not needed, return
			if self.RhoEst.MeasSim.rho_sub_experiments == 0:
				return

			#If the bootstrapped probability list is empty then bootstrapping not ready
			if len(self.RhoEst.MeasSim.boot_probs) == 0:
				raise TypeError("An initial Rho-hat, as well as as bootstrap " +
								"measurements need to be made before attempting" +
								"linear inversion for bootstrap samples.")

			#Iterate through each bootstrap experiment
			#Progressbar goes here
			print("\nConducting bootstrapping. See progress below:")
			bar = progressbar.ProgressBar( maxval=len(self.RhoEst.MeasSim.boot_probs)).start()
			for boot_prob_index in range(len(self.RhoEst.MeasSim.boot_probs)):

				#Start date and time metadata of execution
				self.start_date = dt.datetime.now().date()
				self.start_time = dt.datetime.now().time()
				utc_start_time = dt.datetime.utcnow()

				#Set boot probabilities
				self.RhoEst.boot_rho_prob = self.RhoEst.MeasSim.boot_probs[boot_prob_index]

				#Re-initialize positivity correction to false
				self.RhoEst.MeasSim.boot_rho_hat_pos_corr = 0

				#Create the new bootstrapped Rho and normalizes initial calculations
				boot_rho = self.RhoEst.MeasSim.renormalize(np.matmul(
													 np.matmul(LA.inv(np.matmul(self.RhoEst.MeasSim.meas_ops.T,
														  			self.RhoEst.MeasSim.meas_ops)),
										 		   		  			self.RhoEst.MeasSim.meas_ops.T),
									 					  self.RhoEst.boot_rho_prob).reshape(
									 					  (self.RhoEst.MeasSim.dims,self.RhoEst.MeasSim.dims)))

				#Check that the bootstrapped rho is a valid density matrix
				self.RhoEst.boot_rho_hat = self.RhoEst.MeasSim.validMatrix(boot_rho, bootstrap = True)

				#Calculate fidelity for the bootstrapped sample and add to coverage list
				self.RhoEst.boot_rho_fid = self.RhoEst.calculateFidelity(self.RhoEst.boot_rho_hat)
				self.RhoEst.boot_fids.append(self.RhoEst.boot_rho_fid)

				#Log-likelihood calculation
				self.RhoEst.boot_rho_log_lik = self.RhoEst.MeasSim.calculateLogLikelihood(self.RhoEst.boot_rho_hat, 
																						  self.RhoEst.boot_rho_prob, 
																						  self.RhoEst.MeasSim.rho_sub_trials)

				#Purity calculation
				self.RhoEst.boot_rho_purity = self.RhoEst.MeasSim.calculatePurity(self.RhoEst.boot_rho_hat)
				self.RhoEst.boot_purities.append(self.RhoEst.boot_rho_purity)

				#If it is valid, add the matrix to the list
				self.RhoEst.boot_rhos.append(boot_rho)

				#End date and time metadata
				self.duration_sec = (dt.datetime.utcnow() - utc_start_time).total_seconds()

				#Add logging record
				self.RhoEst.BootLog.addBootRecord(self.RhoEst)

				bar.update(boot_prob_index + 1)

	###########################################################################
	# Public methods enabling flexible solver compatibility
	###########################################################################

	def setEstimator(self, rho_estimator):
		"""
		Attributes the relevant RhoEstimator class to the LinearInversion solver. 

		Args:
			rho_estimator		RhoEstimator class object with all measurement data

		"""
		self.RhoEst = rho_estimator

	