###########################################################################
#
# Diluted MLE Optimizer
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

#Package specific imports
from logger import Log

###########################################################################
# Class and constructor
###########################################################################

class DilutedMLE():
	"""
	One of potentially many solver classes. Operates within the RhoEstimator
	class with shell functions. Metadata about the execution is collected and
	rho_hat is estimated with RrhoR iterative algorithm.

	Attributes:

				### Information pertaining to class object interation ###

			RhoEst 						RhoEstimator class object containing all necessary
										data for analysis.

				### Information pertaining to class execution ###

			RrhoR_iters:				Number of iterations before finding MLE optimum for initial rho_hat
			MLE_timeout:				Boolean indicating if a sample timed out for RrhoR iterations
			boot_RrhoR_iters: 			Number of iterations before finding MLE optimum for boot_rhos
			stopping_crit:				Threshold for error at which optimization stops 
			epsilon:					Percentage of R used in analysis (lower epsilon = more dilution)
			solver_method:				Informs the logger what type of solver was explicitly used
			iter_bound:					Upper bound on iterations to precent timeout
			verbose_exp:				Verbose experiment list... which iterations should have their data logged
										(FOR MLE EXECUTIONS ONLY!)
			verbose_log_filename:		File name of the verbose log created.

	"""

	def __init__(self, stopping_crit = 1e-10,
					   iter_bound = 50000,
					   verbose_exp = []):
		"""
		Constructor for LinearInversion class object. No parameters are given. 
		Necessary information from the estimator class is provided later in the
		execution by QST_sim.

		Args:
			stopping_crit:			Threshold for error at which optimization stops 		
			iter_bound:				kwarg that causes the execution to timeout after 
									a certain number of RrhoR iterations.
			verbose_exp:			Short for verbose experiments, and defaulted to 
									an empty list. Will create verbose logs for MLE
									execution data / metadata for the given experiment index.
		"""
		self.RhoEst = None
		self.RrhoR_iters = 0
		self.MLE_timeout = 0
		self.boot_RrhoR_iters = []
		self.stopping_crit = stopping_crit
		self.epsilon = 1.0
		self.solver_method = "Diluted_MLE"
		self.iter_bound = iter_bound
		self.verbose_exp = verbose_exp
		self.verbose_log_filename = "-"


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
		self.MLEOrch(bootstrap)

	def MLEOrch(self, bootstrap = False):
		"""
		Orchestration package for MLE optimization. Dynamically determines 
		if bootstrapping is occurring or not. For both, it iteratively converges
		on the rho estimate and terminates after the stopping criteria value drops
		below a certain threshold.

		Args:
			bootstrap:			Boolean that indicates whether or not to bootstrap

		"""
		if (not bootstrap):
			#Number of total measurements made across all operators
			#Appropriate probability sample
			trials = self.RhoEst.MeasSim.rho_hat_trials
			probs = self.RhoEst.MeasSim.rho_hat_probs

			#Determine rho_hat and metadata with optimization engine
			rho_hat, RrhoR_iter = self.__optimize_rho(probs, trials)

			#Assign variables after determining if matrix is valid
			self.RhoEst.rho_hat = self.RhoEst.MeasSim.validMatrix(
								  self.RhoEst.MeasSim.imaginaryCorrection(rho_hat))

			#Fidelity calculation
			self.RhoEst.rho_hat_fid = self.RhoEst.calculateFidelity(self.RhoEst.rho_hat)

			#Log-likelihood calculation
			self.RhoEst.rho_hat_log_lik = self.RhoEst.MeasSim.calculateLogLikelihood(self.RhoEst.rho_hat, probs, trials)

			#Purity calculation
			self.RhoEst.rho_hat_purity = self.RhoEst.MeasSim.calculatePurity(self.RhoEst.rho_hat)

			#Assign RrhoR iterative metadata
			self.RrhoR_iters = RrhoR_iter

		#If we are bootstrapping
		else:


			#Number of total measurements made across all operators
			trials = self.RhoEst.MeasSim.rho_sub_trials
			probs = self.RhoEst.MeasSim.boot_probs
			experiments = self.RhoEst.MeasSim.rho_sub_experiments

			#If bootstrapping not needed, return
			if experiments == 0:
				return

			#Iterate through each bootstrap experiment
			#Progressbar goes here
			print("\nConducting bootstrapping. See progress below:")
			bar = progressbar.ProgressBar(maxval=experiments).start()
			for experiment in range(experiments):

				##########################################################
				# Bootstrap loop -- Important for boot_log data collection
				##########################################################

				#Start date and time metadata of execution
				self.start_date = dt.datetime.now().date()
				self.start_time = dt.datetime.now().time()
				utc_start_time = dt.datetime.utcnow()

				#Re-initialize positivity correction to false
				self.RhoEst.MeasSim.boot_rho_hat_pos_corr = 0

				#Set temporary boot probability variable
				self.RhoEst.boot_rho_prob = probs[experiment]

				#If verbose data collection necessary
				if experiment in self.verbose_exp:
					boot_rho, self.boot_RrhoR_iter = self.__optimize_rho(self.RhoEst.boot_rho_prob, trials, verbose = True)

				#If verbose data collection not necessary
				else:
					boot_rho, self.boot_RrhoR_iter = self.__optimize_rho(self.RhoEst.boot_rho_prob, trials)

				#Add information about bootstrapped sample after determining if matrix is valid
				self.RhoEst.boot_rho_hat = self.RhoEst.MeasSim.validMatrix(
										   self.RhoEst.MeasSim.imaginaryCorrection(boot_rho))

				#Add rho_hat to bootstrapped list
				self.RhoEst.boot_rhos.append(self.RhoEst.boot_rho_hat)

				#Add metadata for relevant bootstrap sample
				self.boot_RrhoR_iters.append(self.boot_RrhoR_iter)

				#Calculate fidelity for the bootstrapped sample and add to coverage array
				self.RhoEst.boot_rho_fid = self.RhoEst.calculateFidelity(boot_rho)
				self.RhoEst.boot_fids.append(self.RhoEst.boot_rho_fid)


				#Log-likelihood calculation
				self.RhoEst.boot_rho_log_lik = self.RhoEst.MeasSim.calculateLogLikelihood(self.RhoEst.boot_rho_hat, 
																						  self.RhoEst.boot_rho_prob, 
																						  trials)

				#Purity calculation
				self.RhoEst.boot_rho_purity = self.RhoEst.MeasSim.calculatePurity(self.RhoEst.boot_rho_hat)
				self.RhoEst.boot_purities.append(self.RhoEst.boot_rho_purity)

				#End date and time metadata
				self.duration_sec = (dt.datetime.utcnow() - utc_start_time).total_seconds()

				#Add logging record
				self.RhoEst.BootLog.addBootRecord(self.RhoEst)

				#Update progressbar
				bar.update(experiment + 1)

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

	###########################################################################
	# Private methods used to optimize rho with Diluted MLE
	###########################################################################

	def __optimize_rho(self, probs, trials, verbose = False):
		"""
		Optimization package to determine rho_hat. Combines helper functions to determine
		R, determine rho_hat, and then repeats, potentially adding dilution if the likelihood
		of the function does not increase.

		Args:
			Probs:				Measurement probabilities, empirically found with random # generation
			trials:				Number of measurement trials for the experiment
			verbose:			kwarg where additional logging may be added if necessary. Defaults to false.

		Returns:
			current_rho:		DilutedMLE optimized rho_hat estimate
			RrhoR_iterations:	Number of iterations until optimization was complete
		"""

		#Reset epsilon
		self.epsilon = 1.0
		self.epsilon_corrections = 0

		#If optimization is verbose, store ALL information
		if verbose:

			#Initialize verbose log and set all log names
			#Must set each log name AFTER initialization for consistency
			self.verbose_log = Log("Master-Log", "Boot-Log", "MLE-Exec-Log")
			self.verbose_log.master_log_name = self.RhoEst.BootLog.master_log_name
			self.verbose_log.boot_log_name = self.RhoEst.BootLog.boot_log_name
			self.verbose_log_filename = self.verbose_log.verbose_log_name

		#Initialize current_rho to be the maximally mixed state
		self.current_rho = (1./self.RhoEst.MeasSim.dims)*(np.identity(self.RhoEst.MeasSim.dims))

		#Make R using the maximally mixed state
		self.R = self.__makeR(self.current_rho, probs, trials)

		#Increment the number of RrhoR iterations
		RrhoR_iterations = 1

		#Get likelihood (slope estimation) measurement
		self.log_likelihood = self.RhoEst.MeasSim.calculateLogLikelihood(self.current_rho, probs, trials)

		#If the execution is verbose, add the first verbose record
		if verbose:

			#Add first Verbose Log record
			self.verbose_log.addVerboseRecord(self)

		#While the likelihood (slope) higher than a ceratin threshold
		while (self.__getLikelihoodSlope(self.R, trials) > self.stopping_crit):

			#Get the upper bound information
			self.upper_bound = self.__getLikelihoodSlope(self.R, trials)

			#Make new rho
			self.current_rho = self.__makeRho(self.R, self.current_rho)

			#Make new R
			self.R = self.__makeR(self.current_rho, probs, trials)

			#If log_likelihood does not improve, adjust epsilon
			if (self.RhoEst.MeasSim.calculateLogLikelihood(self.current_rho, probs, trials) <= self.log_likelihood):

				#Increment epsilon corrections
				self.epsilon_corrections += 1

				#Reduce epsilon to be 3/4 of its prior amount
				self.epsilon = (self.epsilon * 0.75)
 
 			#Re-assign log_likelihood
			self.log_likelihood = self.RhoEst.MeasSim.calculateLogLikelihood(self.current_rho, probs, trials)

			#Update the RrhoR iterations
			RrhoR_iterations += 1

			#If execution is verbose, add a verbose records
			if verbose:
				self.verbose_log.addVerboseRecord(self)

			#If the iteration count exceeds a bound, set the execution
			if RrhoR_iterations >= self.iter_bound:

				#If execution is verbose, save the verbose log
				if verbose:
					self.verbose_log.saveVerboseLog()

				#Return the current rho and iterations on timeout
				#Note that the iteration timed out
				self.MLE_timeout = 0

				return self.current_rho, RrhoR_iterations

		#If no timeout and verbose, save verbose log
		if verbose:
			self.verbose_log.saveVerboseLog()

		#Return the current_rho estimate and iterations
		return self.current_rho, RrhoR_iterations

			
	def __makeR(self, current_rho, probs, trials):
		"""
		Create R for R-rho-R optimization for MLE. Generated from experimentally collected data.
		This is dynamic and can be run for both bootstrap and non-bootstrap samples.

		Uses formula found in:
		"Glancy, S., Knill, E., & Girard, M. (2012). Gradient-based stopping rules 
		for maximum-likelihood quantum-state tomography. New Journal of Physics, 14(9), 095017."

		Args:
			current_rho:		rho_hat currently found by iterative optimization
			probs:				Empirically determined probability array
			

		Dependencies:
			tomoSim must be run such that probabilistic data is collected and can be used to create R

		Returns:
			R:					Matrix that can be used to optimize rho_hat
		"""

		#Initialize number of trials
		N = trials

		#Initialze R variable
		R = 0

		#Iterate through all the measurement bases
		for op_index in range(0, self.RhoEst.MeasSim.meas_ops.shape[0], 2):

			#Re-shape pi_zero and pi_one
			pi_zero = self.RhoEst.MeasSim.meas_ops[op_index].reshape((self.RhoEst.MeasSim.dims,self.RhoEst.MeasSim.dims), order = "F")
			pi_one = self.RhoEst.MeasSim.meas_ops[op_index+1].reshape((self.RhoEst.MeasSim.dims,self.RhoEst.MeasSim.dims), order = "F")

			#Find pi_zero and pi_one frequencies for the experiment
			pi_zero_freq = float((probs[op_index] + (1 - probs[op_index + 1]))*N)
			pi_one_freq = float(((1 - probs[op_index]) + (probs[op_index + 1]))*N)

			#Initialize trace_pi_zero and trace_pi_one
			trace_pi_zero = 0
			trace_pi_one  = 0
			
			#Check and see if there are div0 errors for the trace, and handle them
			try:
				trace_pi_zero = (1/float((np.matmul(current_rho, pi_zero)).trace().real))
				trace_pi_one = (1/float((np.matmul(current_rho, pi_one)).trace().real))

			#Exceptions currently ignored, but this can be logged if necessary
			except:
				pass

			#Add pi_zero component to the sum
			R += pi_zero_freq * trace_pi_zero * (pi_zero)

			#Add pi_one component to the sum
			R += pi_one_freq * trace_pi_one * (pi_one)

		#Return linear combination of R and the identity matrix (via epsilon)
		return self.epsilon*R + (1 - self.epsilon)*np.identity(self.RhoEst.MeasSim.dims)

	def __makeRho(self,R,rho_hat_prev):
		"""
		Make new Rho matrix by conducting iterative R-Rho-R analysis

		Args:
			R:					R matrix computed from empirical data
			rho_hat_prev:		Rho_hat to be used to create new rho in this iteration
								Note: it is likely that the first iteration of this process will
									  make use of a rho_hat that is in a maximally mixed state. 

		Returns:
			new_rho				New rho calculated using formula above and normalized.
		"""

		new_rho = self.RhoEst.MeasSim.renormalize(np.matmul(R, np.matmul(rho_hat_prev, R)))

		return new_rho

	def __getLikelihoodSlope(self, R, trials):
		"""
		Provide an approximation for the slope of the concave curve whose
		maximum is the optimization point. When the slope approaches zero
		past a certain threshold, the execution terminates. The formula 
		and stopping criteria can be found in the following paper:

		"Glancy, S., Knill, E., & Girard, M. (2012). Gradient-based stopping rules 
		for maximum-likelihood quantum-state tomography. New Journal of Physics, 14(9), 095017."

		Args:
			R:			R matrix determined form Rho to complete MLE optimization
			trials:		Number of trials in the particular experiment

		Returns:
			slope_est:	Slope estimation used as a proxy for likelihood. The close
						the slope is to zero the more optimal the solution
		"""

		#Get the eigenvalues
		eigvals = LA.eigvalsh(R)

		#Return the likelihood of the function (slope estimation of concave function)
		slope_est = max(eigvals) - self.RhoEst.MeasSim.meas_ops.shape[0]*trials

		return slope_est


		

	
