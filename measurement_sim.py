###########################################################################
#
# Quantum State Tomography Simulation Orchestrator
# Author: Sam Showalter
# Date: June 5, 2018
#
###########################################################################


###########################################################################
# Module and library imports 
###########################################################################

#Logistic and system-based imports
import sys

#Data analysis and scientific computing libraries
import numpy as np 
from numpy import linalg as LA
from scipy.linalg import sqrtm

###########################################################################
# Class and Constructor
###########################################################################

class MeasSim():

	"""
	Performs quantum state tomography.

	Attributes:

					### Information pertaining to class execution ###

		rho_goal:						The target (or goal) density matrix
		noise_perc:						Amount of noise to be included in creating rho_true
										rho_true is a physical representation of rho_goal
										in the real world, and incorporates a specified
										level of noise from a maximally mixed state

		initial_rho_hat_trials:			Number of measurement trials to estimate initial rho_hat
		rho_sub_experiments:			Number of bootstrapped rhos to be created from initial rho_hat
		rho_sub_trials:					Number of measurement trials to estimate the bootstrapped rho_hats
		rho_true:						Linear combination of rho_goal and maximally mixed state
										determined by noise_perc
		dim:							Dimension of Hilbert space
		tol:							Noise tolerance for measurement
		seed:							Random seed for random number measurement execution
		measurement_ops:				Provided. Articulates the six basis states in the blochsphere

					### Information calculated during class execution ###

		probs:							Expected probabilities to be found in measurement. 
										Calculated from Rho_True
		boot_probs:						Expected probabilities to be found in bootstrap measurement.
										Calculated from Rho_Hat

					### Metadata information for Logging ###

		rho_goal_pos_corr:				Positivity correction boolean for rho_goal
		rho_hat_pos_corr:				Positivity correction boolean for rho_hat
		boot_rho_hat_pos_corr_count:	Positivity correction boolean count for bootstraps
		boot_rho_hat_pos_corr:			(Temporary) boolean for specific bootstrap iteration (fine-tunes logging)


	"""

	def __init__(self,
				 rho_goal, 
				 noise_perc,
				 initial_rho_hat_trials,
				 rho_sub_experiments,
				 rho_sub_trials,
				 file_data = None,
				 flat_file = False,
				 tol = 1e-7,
				 measurement_ops = np.matrix(#Computational basis
											[np.array([1, 0, 0, 0]),
											 np.array([0, 0, 0, 1]),

											 #Right left basis
											 0.5*np.array([1, 1, 1, 1]),
											 0.5*np.array([1, -1, -1, 1]),

											 #Plus- minus basis
											 0.5*np.array([1, 0+1j, 0-1j, 1]),
											 0.5*np.array([1, 0-1j, 0+1j, 1])]
											)):

		"""
		Initialize tomographer with supplied paramters.

		Args:
			rho_goal:					Goal or target density matrix we would like to approximate 
			noise_perc:					Percentage of maximally mixed state to add to rho_true
			initial_rho_hat_trials:		Number of trials for measurements in each basis to create initial rho_hat
			rho_sub_experiments:		Number of additional datasets to create, along with rho_hats and fidelities
			rho_sub_trials:				Number of trials to give for each sub_experiment; size of each dataset for 
										sub experiments
			tol:						Noise tolerance for measurement
			seed:						Random seed for random number measurement execution					
			measurement_ops:			Provided. Articulates the six basis states in the blochsphere
		"""

		#Establish number of dxd dimensions in Hilbert space
		self.dims = rho_goal.shape[0]

		#Discover the seed (May be set or not)
		#If the number is over 100000, then it likely
		#was not set.
		self.seed = np.random.get_state()[1][0]

		#Set noise tolerance for measurement
		self.tol = tol

		#Rho_hat matrix corrections for rho_hat 
		#initial as well as bootstrapped rho_hats
		#Corrections made to coerce positivity
		# 0 = False, 1 = True
		self.rho_goal_pos_corr = 0
		self.rho_hat_pos_corr = 0
		self.boot_rho_hat_pos_corr_count = 0
		self.boot_rho_hat_pos_corr = 0

		#Maximally mixed state for a quantum bit of "dim" dimensions
		max_mixed_state = np.identity(self.dims)/self.dims
		
		#Initialize rho_goal and noice_percent
		self.rho_goal = rho_goal
		self.noise_perc = noise_perc
		
		# Input validation for rho true and rho_goal.
		# Purity check for rho_goal
		self.rho_goal = self.validMatrix(self.rho_goal, rho_goal = True)
		self.rho_goal_purity = self.calculatePurity(self.rho_goal)

		#Creation of rho_true and validation of it
		#Purity check for rho_true
		self.rho_true = (1-noise_perc)*self.rho_goal + (noise_perc*max_mixed_state)
		self.rho_true = self.validMatrix(self.rho_true)
		self.rho_true_purity = self.calculatePurity(self.rho_true)

		#Initialize iteration and experiment logistics
		self.rho_hat_trials = initial_rho_hat_trials
		self.rho_sub_experiments = rho_sub_experiments
		self.rho_sub_trials = rho_sub_trials

		#Initialize measurement operations
		self.meas_ops = measurement_ops

		# #Chang meas_ops if dims > 1
		if self.dims == 4:
			self.meas_ops = np.matrix(np.kron(self.meas_ops,self.meas_ops))

		#Determine the probabilities that one would expect to see using rho_true
		self.probs = self.meas_ops*np.reshape(self.rho_true,(self.rho_true.size,1))

		# Miscellaneous variables pertaining to initial
		# Rho hat probabilities
		self.rho_hat = None 				#Kept only for error handling purposes
		self.rho_hat_probs = None

		#Repositories for the bootstrap rhos
		self.boot_probs = []

		#################################################################################
		# DEBUGGING -- can be commented out whenever
		#################################################################################

		# print("Dims: %s\n"%(self.dims))
		# print("Rho goal \n%s\n"%self.rho_goal)
		# print("Rho true \n%s\n"%self.rho_true)

		# #Measurement operations
		# print("Measurement operations:")
		# print(self.meas_ops)

		# #Print eigenvalues of matrix
		# w = LA.eigvalsh(self.rho_true)
		# print("\nEigenvalues of rho_true")
		# print(w)

		# #Expected probabilities to see
		# print("\nProbabilies of each outcome:")
		# print(self.probs)

		# #Properly reshape the density matrix to a column vector
		# print("\nColumn-wise density matrix.")
		# print(np.reshape(self.rho_true,(self.rho_true.size,1)))

		###################################################################################


	###########################################################################
	# Orchestrator
	###########################################################################

	def measSim(self, bootstrap = False):
		"""
		Initialize tomographer with supplied paramters. Dynamic for both initial rho hat
		tomography as well as for bootstrap sample bulk tomography.

		Args:
			bootstrap:			Whether or not this is a bootstrap sample

		Generates:
			(#-meas ops. x 1) matrix with average, empirically found probabilities of measurement.
		"""

		#Outer loop logic for using Rho_true
		if not bootstrap:
			trials = self.rho_hat_trials
			probs = self.probs
			experiments = 1

		#Inner loop logic for using Rho_hat
		else:

			#Check to be sure that rho_hat has already been determined
			if self.rho_hat is None:
				raise ValueError("Initial tomography must be run to generate" +
								 "an initial rho_hat before bootstrapping may begin.")

			#Assign appropriate inner loop bootstrapping parameters
			trials = self.rho_sub_trials
			probs = self.meas_ops*np.reshape(self.rho_hat,(self.rho_hat.size,1))
			experiments = self.rho_sub_experiments

		#Loop for sub_rho-hat experiments
		for experiment in range(experiments):

			#Generate matrix of random numbers to use as measurement data
			raw_trial_data = np.random.rand(probs.shape[0],trials)

			#Convert raw random values into either a zero or one
			final_trial_data = (raw_trial_data < probs)

			#Generate frequencies for each measurement operator
			freqs = final_trial_data.mean(1)

			# Encapsulate empirical probs in Tomography object
			# for outer loop discovery of initial rho_hat
			if not bootstrap:

				#Assign probability measurements for rho_hat
				self.rho_hat_probs = freqs

				#Calculate likelihood for rho_true
				self.rho_true_log_lik = self.calculateLogLikelihood(self.rho_true,
																	self.rho_hat_probs,
																	trials)

			#If not using the initial Rho, then run the Bootstrapping subsample information
			else:
				self.boot_probs.append(freqs)


		#Print statements for debugging or visualization
		###############################################################################
		# print("\nRaw trial data")
		# print(raw_trial_data)

		# print("\n Final trial data")
		# print(final_trial_data)

		# print("\nMean result for empirical probabilities")
		# print(empirical_probs)
		###############################################################################
	
	#############################################################################
	# Public methods for error handling and matrix adjustment across the project
	#############################################################################

	def calculatePurity(self, matrix):
		"""
		Determines the purity of the matrix by squaring the
		matrix and taking its trace.

		Args:
			matrix:					Input matrix for which purity is found

		Returns:
			purity:					Purity of the density matrix
		"""

		purity = float(np.matmul(matrix,matrix).trace().real) 

		return purity

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
		for op_index in range(0, self.meas_ops.shape[0], 2):

			#Re-shape pi_zero and pi_one
			pi_zero = self.meas_ops[op_index].reshape((self.dims,self.dims), order = "F")
			pi_one = self.meas_ops[op_index+1].reshape((self.dims,self.dims), order = "F")

			#Find pi_zero and pi_one frequencies for the experiment
			pi_zero_freq = float((probs[op_index] + (1 - probs[op_index + 1]))*N)
			pi_one_freq = float(((1 - probs[op_index]) + (probs[op_index + 1]))*N)

			#Initialize trace_pi_zero and trace_pi_one
			ln_trace_pi_zero = 0
			ln_trace_pi_one  = 0
			
			#Handle division by zero errors
			with np.errstate(divide='raise'):

				#Try to formulate a value
				try:
					#Get natural log of the trace for pi zero and pi one
					ln_trace_pi_zero = np.log(float((np.matmul(rho, pi_zero)).trace().real))
					ln_trace_pi_one = np.log(float((np.matmul(rho, pi_one)).trace().real))

				#Catch any undefined warnings or exceptions
				except:
					pass

			#Add pi_zero component to the sum
			log_likelihood += pi_zero_freq * ln_trace_pi_zero

			#Add pi_one component to the sum
			log_likelihood += pi_one_freq * ln_trace_pi_one

		#Return linear combination of R and the identity matrix (via epsilon)
		return log_likelihood/N

	def validMatrix(self, matrix, bootstrap = False, rho_goal = False):
		"""
		Input parameter validation for rho goal and rho true matrices (though only rho_goal necessary)
		Includes conditional parameters for bootstrap samples. If "Matrix is not positive; it has negative eigenvalues",
		then positivityCorrection is run to adjust for this. These occurrences are logged for posterity.

		Args:
			Matrix: 				Either Rho_goal or Rho_true, both found or derived from input params
			bootstrap:				Whether or not this is a bootstrapped sample
			rho_goal:				Boolean informing execution if the matrix is rho_goal (for metadata)

		Raises:
			ValueError:				Matrix does not have a valid trace
			ValueError:				Matrix is not Hermitian
			ValueError:				Rho goal is not a positive matrix
			
		"""
		#Verifies matrix is of the correct data type
		matrix = np.matrix(matrix)

		#Check to see if the trace equals 1
		#Use approximation with arbitrary smallness due to Python's float inconsistencies
		if np.abs(matrix.trace() - 1) > self.tol:
			raise ValueError("Rho goal input does not have a "
                                "valid trace that sums to 1.")

		#Check to see if the matrix is hermitian
		if not np.allclose(matrix.H, matrix):
			raise ValueError("Matrix input is not Hermitian "
                                "and therefore not valid for tomography.")

		#Check to see if the Eigenvalues are positive
		eigenvals = LA.eigvalsh(matrix)
		for val in eigenvals:
			if val < 0:
				#Return a positivity-corrected matrix
				if rho_goal:
					#If rho_goal is not a positive matrix
					raise ValueError("Non-positive rho_goal input matrix. " +
									 "Please check input parameters.")

				#If it is another matrix	
				else:
					matrix = self.positivityCorrection(matrix, bootstrap)

		#Must double check data type because of numpy quirck to transform things into arrays
		return np.matrix(matrix)


	def renormalize(self, matrix):
		"""
		Transform a matrix by a scalar value element-wise such that the trace of the matrix is one.
		This is necessary when conducting MLE calculations

		Args:
			matrix:				Input matrix to be returned normalized per the definition above

		Returns:
			norm_matrix			Matrix that has been normalized
		"""

		#Reference value for the trace of the matrix
		trace_ref = matrix.trace()

		#Make sure you are just getting real values in float form
		transform_scalar =  float((1/(trace_ref)).real)

		#Generate the normal matrix
		norm_matrix = matrix * transform_scalar

		return norm_matrix

	def positivityCorrection(self, matrix, bootstrap = False):
		"""
		Corrects negative matrices to be positive, valid density operators. It does this by
		decomposing the matrix to its eigenvalues and eigenvectors, and removing any negative
		eigenvalue-vector components.

		Args:
			matrix:				Input matrix to be converted to a positive matrix
			bootstrap:			Boolean determiner of bootstrapping

		Returns:
			corrected_rho		Positive matrix corresponding to the negative input matrix
		"""

		#Get eigenvalues and eigenvectors of the matrix (MUST use Hermitian function eigh)
		eigvals, eigvecs = LA.eigh(matrix)

		#Initialize corrected Rho variable
		correctedRho = 0

		#Re-compose the density matrix
		for eig_index in range(len(eigvals)):

			#Add eigenvalue-vector component if eigenvalue is positive, else add nothing.
			correctedRho += np.matrix(max(eigvals[eig_index], 0) * np.outer(eigvecs[:,eig_index], eigvecs[:,eig_index].getH()))

		#Metadata logging logic to determine how often matrices are not positive
		# or if they are positive at all. (1 = True, 0 = False)
		if bootstrap:
			self.boot_rho_hat_pos_corr_count += 1
			self.boot_rho_hat_pos_corr = 1
		else:
			self.rho_hat_pos_corr = 1

		#Return normalized, correctedRho
		return self.renormalize(correctedRho)

	def imaginaryCorrection(self,matrix):
		"""
		Corrects matrices for any imaginary components created as the result of a 
		rounding error in Python. Function checks to determine whether or not each imaginary
		component is trivial at a certain threshold. If the term is not, then an
		error is thrown.

		Args:
			matrix:				Input matrix to be checked for imaginary values

		Raises:
			ValueError:			Raises an error if the trace of the matrix has non-trivial
								imaginary elements.

		Returns:
			matrix				Matrix with non-imaginary trace
		"""

		#For each value in the index, check its imaginary component
		for trace_index in range(matrix.shape[0]):

			#Examine the element's imaginary component
			if matrix[trace_index, trace_index].imag < self.tol:
				matrix[trace_index, trace_index] = matrix[trace_index, trace_index].real

			#If the imaginary component of the matrix is non-trivial
			else:
				raise ValueError("Matrix trace has non-trivial imaginary elements. " +
								 "Please check your calculations.")

		return matrix

