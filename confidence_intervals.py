###########################################################################
#
# Quantum State Tomography Simulation Orchestrator
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

#Data analysis and scientific computing libraries
import numpy as np 
import pandas as pd 
from numpy import linalg as LA
from scipy.stats import sem, norm
from scipy.linalg import sqrtm

###########################################################################
# Class and Constructor
###########################################################################

class Coverage():
	"""
	Coverage class that provides a dynamic analysis of
	bootstrapped fidelity samples using a variety of 
	confidence intervals.

	Attributes:

				### Information pertaining to class object interation ###

		RhoEst:					Relevant RhoEstimator class with all relevant data

				### Information pertaining to class execution ###

		confidence: 			Confidence level
		alpha:					1 - confidence level
		sided: 					The type of confidence interval requested
		method:					Types of confidence intervals requested (can be "all" or a specific interval)
		intervals:				Dictionary for fidelity and purity confidence intervals
		perc_bounds:			Percentage bounds determined by "sided" parameter for CI
		type:					Type of confidence interval(s) requested
	"""

	def  __init__(self,
				  confidence = 0.95,
				  interval_type = "all",
				  sided = "two_sided",
				  method = "all"):
		"""
		Constructor for Coverage class. Defaults the RhoEst
		class to None, as it is assigned by the QST_sim 
		orcheestrator. All other variables are either attributed
		to the class or defaulted to zero

		Args:
			confidence 			Confidence level
			interval_type		The subject of the interval (fidelity, purity,...)
			sided 				The type of confidence interval requested
			methods				Types of confidence intervals requested
		"""

		self.RhoEst = None
		self.confidence = confidence
		self.alpha = 1 - confidence 
		self.sided = sided 
		self.method = method
		self.type = interval_type

		#Get percent bounds based on the kind of test
		self.perc_bounds = self.__getPercBounds()

		#purity bootstrap interval dictionary
		#initialized to dashes for logging purposes
		self.intervals = {"fidelity": {    "percentile":		("-","-"),
										   "normal":			("-","-"),
										   "basic":				("-","-"),
										   "bias_corrected":	("-","-"),
										   "BC_a":				("-","-")},

						  "purity":   {    "percentile":		("-","-"),
										   "normal":			("-","-"),
										   "basic":				("-","-"),
										   "bias_corrected":	("-","-"),
										   "BC_a":				("-","-")}}


	###########################################################################
	# Orchestrator
	###########################################################################

	def coverage_orch(self):
		"""
		Coverage orchestrator for creating confidence intervals. 

		For any and all methods requested, this function will call all the
		functions called from the "interval_roster" function dictionary.

		Raises:
			ValueError:					If interval keyword input is not
										recognized.
		"""

		#Dictionary of confidence interval functions and their keywords
		interval_roster = {"percentile":		self.CI_percentile,
						   "normal":			self.CI_norm_dist,
						   "basic":				self.CI_basic,
						   "bias_corrected":	self.CI_bias_corrected,
						   "BC_a":				self.CI_BC_a}

		type_roster = {"fidelity":				self.RhoEst.boot_fids,
					   "purity":				self.RhoEst.boot_purities}

		#Initialize key list
		key_list = []

		#Determine what kind of types are needed for the confidence intervals
		if self.type == "all":
			key_list = type_roster.keys()
		else:
			key_list = [self.type]

		#Iterate through each type of interval (fidelity, purity, etc.)
		for type_key in type_roster:

			#Run the confidence interval methods for a specific type
			if type_key in key_list:


				#If every confidence interval is requested
				if self.method == "all":

					#Iterate through interval roster function dictionary
					for key in interval_roster:

							#Iterate through each key and get results
							self.intervals[type_key][key] = interval_roster[key](type_roster[type_key], type_key)
				else:
					#General try-catch for specific interval
					try:
						self.intervals[type_key][key] = interval_roster[self.method](type_roster[type_key], type_key)

					#Raise keyword error if specific interval not recognized
					except:
						raise KeyError("Interval keyword not recognized. Check inputs.")

		#Debugging: shows all the intervals
		#print(self.fid_intervals)

	###########################################################################
	# Confidence Intervals
	###########################################################################

	def CI_percentile(self, sample, data_type):
		"""
		Confidence interval created with the percentile method. 

		Args:
			sample:				Bootstrap sample of fidelities
			data_type:				Type of data. Un-used argument here as of now.

		Returns:
			conf_interval:		Confidence interval, normal dist method
			
		"""
		
		#Sort the bootstrapped sample of estimates
		sort_sample = sorted(sample)

		#Initialize the upper and lower bounds
		b_l = 0
		b_u = 0

		# Determine if the percentile is an integer or not.
		# If not, average the two proximate estimates
		if not isinstance(self.perc_bounds[0]*self.RhoEst.MeasSim.rho_sub_experiments, int):
			b_l = (sort_sample[int(np.ceil(self.perc_bounds[0]*self.RhoEst.MeasSim.rho_sub_experiments)) - 1] +
				  sort_sample[int(np.floor(self.perc_bounds[0]*self.RhoEst.MeasSim.rho_sub_experiments)) - 1])/2
		else:
			b_l = sort_sample[self.perc_bounds[0]*self.RhoEst.MeasSim.rho_sub_experiments - 1]

		if not isinstance(self.perc_bounds[1]*self.RhoEst.MeasSim.rho_sub_experiments, int):
			b_u = (sort_sample[int(np.ceil(self.perc_bounds[1]*self.RhoEst.MeasSim.rho_sub_experiments)) - 1] +
				  sort_sample[int(np.floor(self.perc_bounds[1]*self.RhoEst.MeasSim.rho_sub_experiments)) - 1])/2
		else:
			b_u = sort_sample[self.perc_bounds[1]*self.RhoEst.MeasSim.rho_sub_experiments - 1]

		# Create confidence interval tuple
		conf_interval = (b_l, b_u)

		return conf_interval


	def CI_norm_dist(self, sample, data_type):
		"""
		Confidence interval created assuming that the
		probability distribution is normal. This was NOT
		asked for... I accidentally misunderstood what was asked.
		May be deleted in future commits.

		Args:
			sample:				Bootstrap sample of fidelities
			data_type:				Type of data. Un-used argument here as of now.

		Returns:
			conf_interval:		Confidence interval, normal dist method
		"""

		#Get the standard deviation
		SE = np.std(sample)

		#Get the mean
		mean = np.mean(sample)

		#Get low high z-scores
		lowZ = norm.ppf(self.perc_bounds[0])	#Low
		highZ = norm.ppf(self.perc_bounds[1])	#High

		#Low and high bound values
		low_bound = mean + SE*lowZ
		high_bound = mean + SE*highZ

		#Confidence interval tuple
		conf_interval = (low_bound, high_bound)

		return conf_interval

	def CI_basic(self, sample, data_type):
		"""
		Non-studentized pivotal confidence interval (basic) method. Determines
		the confidence interval based on the errors between the initial estimate
		and the bootstrapped estimates. 

		Args:
			sample: 			Sample of bootstrapped values of a certain type
			data_type:			Type of data. 
		"""

		#Establish reference dictionary
		ref_dict = {"fidelity":	self.RhoEst.rho_hat_fid,
					"purity":	self.RhoEst.rho_hat_purity}

		#Determine reference data balue
		ref = ref_dict[data_type]

		#Initialize lower and upper bounds for errors
		e_l = 0
		e_u = 0

		#Get errors
		errors = sorted([(boot_val - ref) for boot_val in sample])

		# Determine if the percentile bound will have an integer index or not
		# If not, take the average to the two proximate error values
		# Must subtract one because of zero indexing
		if not isinstance(self.perc_bounds[0]*self.RhoEst.MeasSim.rho_sub_experiments, int):
			e_l = (errors[int(np.ceil(self.perc_bounds[0]*self.RhoEst.MeasSim.rho_sub_experiments)) - 1] +
				  errors[int(np.floor(self.perc_bounds[0]*self.RhoEst.MeasSim.rho_sub_experiments)) - 1])/2
		else:
			e_l = errors[self.perc_bounds[0]*self.RhoEst.MeasSim.rho_sub_experiments - 1]

		if not isinstance(self.perc_bounds[1]*self.RhoEst.MeasSim.rho_sub_experiments, int):
			e_u = (errors[int(np.ceil(self.perc_bounds[1]*self.RhoEst.MeasSim.rho_sub_experiments)) - 1] +
				  errors[int(np.floor(self.perc_bounds[1]*self.RhoEst.MeasSim.rho_sub_experiments)) - 1])/2
		else:
			e_u = errors[self.perc_bounds[1]*self.RhoEst.MeasSim.rho_sub_experiments - 1]

		#Construction of basic confidence interval
		conf_interval = (ref - e_u, ref - e_l)

		return conf_interval

	def CI_bias_corrected(self, sample, data_type):
		"""
		Bias corrected confidence interval function. Uses the bias correction
		term z0 to improve the interval bounds. This function is derived from the
		following paper:

		"Carpenter, J., & Bithell, J. (2000). Bootstrap confidence intervals: when, which, what? A practical guide for medical statisticians."

		Args:
			sample:				Sample of rho_sub_i_hat_fid estimates
			data_type:			Type of data. Un-used argument here as of now.

		Returns:
			s_x_i:				s(x_(i)), or the deviations from the mean of 
								the bootstrapped sample observations.
		"""

		#Sort the sample
		sort_sample = sorted(sample)

		#Get bias term, z0
		z0 = self.__z_0(sort_sample)

		# #Printing for debugging purposes
		# print("\nZ0: \n%s\n"%z0)

		#Get rid of the plus one for indexing purposes
		Q_l = ((self.RhoEst.MeasSim.rho_sub_experiments + 1) *
			 norm.cdf(2*z0 + norm.ppf(self.perc_bounds[0])))

		# #Printing for debugging purposes
		# print("\nQ_l: \n%s\n"%Q_l)

		Q_u = ((self.RhoEst.MeasSim.rho_sub_experiments + 1) *
			 norm.cdf(2*z0 + norm.ppf(self.perc_bounds[1])))

		# #Printing for debugging purposes
		# print("\nQ_l:")
		# print(Q_l)

		# print("\nQ_u:")
		# print(Q_u)

		# Determine if the percentile bound will have an integer index or not
		# If not, take the average to the two proximate values
		# Must subtract one because of zero-based indexing
		if not isinstance(Q_l*self.RhoEst.MeasSim.rho_sub_experiments, int):
			b_l = (sort_sample[int(min(max(np.ceil(Q_l),1),self.RhoEst.MeasSim.rho_sub_experiments) - 1)] +
				  sort_sample[int(min(max(np.floor(Q_l),1),self.RhoEst.MeasSim.rho_sub_experiments) - 1)])/2

		else:
			b_l = sort_sample[min(max(1, Q_l), self.RhoEst.MeasSim.rho_sub_experiments) - 1]

		if not isinstance(Q_u*self.RhoEst.MeasSim.rho_sub_experiments, int):
			b_u = (sort_sample[int(max(min(self.RhoEst.MeasSim.rho_sub_experiments, np.ceil(Q_u)), 1)) - 1] +
				  sort_sample[int(max(min(self.RhoEst.MeasSim.rho_sub_experiments, np.floor(Q_u)), 1)) - 1])/2
		else:
			b_u = sort_sample[max(min(self.RhoEst.MeasSim.rho_sub_experiments, Q_u), 1) - 1]

		#Make confidence interval
		conf_interval = (b_l, b_u)

		return conf_interval

	def CI_BC_a(self, sample, data_type):
		"""
		Bias corrected and accelerated method for determining a 
		bootstrapped confidence interval. Function adjusts traditional 
		percentile bounds with both a bias term z0 and acceleration acc. 
		Function taken from: 

		"Efron, B., & Tibshirani, R. J. (1994). An introduction to the bootstrap. CRC press."
		pg. 185

		Args:
			sample:				Sample of rho_sub_i_hat_fid estimates
			data_type:			Type of data. Un-used argument here as of now.

		Returns:
			conf_interval:		Confidence interval
		"""

		#Give CI name
		CI_name = "BC_a"

		#Sort the sample of estimates
		sort_sample = sorted(sample)

		#Determine bias term z0 and accleration
		z0 = self.__z_0(sort_sample)
		acc = self.__acceleration(sort_sample)

		#Print statements for debugging
		# print("\nZ0: \n%s\n"%z0)
		# print("Acc: \n%s\n"%acc)

		#Dervie adjusted upper and lower percentile bounds
		adj_low_bound = norm.cdf(z0 + ((z0 + norm.ppf(self.perc_bounds[0]))/
										(1 - acc*(z0 + self.perc_bounds[0]))))

		adj_upper_bound = norm.cdf(z0 + ((z0 + norm.ppf(self.perc_bounds[1]))/
										(1 - acc*(z0 + self.perc_bounds[1]))))

		#Print statements for debugging purposes
		# print("\nadj_low_bound: \n%s\n"%adj_low_bound)
		# print("\nadj_high_bound: \n%s\n"%adj_upper_bound)

		# Determine if the percentile bound will have an integer index or not
		# If not, take the average to the two proximate values
		# Must subtract one because of zero indexing
		if not isinstance(adj_low_bound*self.RhoEst.MeasSim.rho_sub_experiments, int):
			b_l = (sort_sample[int(min(max(np.ceil(adj_low_bound*self.RhoEst.MeasSim.rho_sub_experiments),1),
										   self.RhoEst.MeasSim.rho_sub_experiments) - 1)] +
				  sort_sample[int(min(max(np.floor(adj_low_bound*self.RhoEst.MeasSim.rho_sub_experiments),1),
				  						   self.RhoEst.MeasSim.rho_sub_experiments) - 1)])/2

		else:
			b_l = sort_sample[min(max(1, adj_low_bound*self.RhoEst.MeasSim.rho_sub_experiments), 
									  self.RhoEst.MeasSim.rho_sub_experiments) - 1]

		if not isinstance(adj_upper_bound*self.RhoEst.MeasSim.rho_sub_experiments, int):
			b_u = (sort_sample[int(max(min(np.ceil(adj_upper_bound*self.RhoEst.MeasSim.rho_sub_experiments), 
										   self.RhoEst.MeasSim.rho_sub_experiments), 1)) - 1] +
				  sort_sample[int(max(min(np.floor(adj_upper_bound*self.RhoEst.MeasSim.rho_sub_experiments),
				  						   self.RhoEst.MeasSim.rho_sub_experiments), 1)) - 1])/2
		else:
			b_u = sort_sample[max(min(adj_upper_bound*self.RhoEst.MeasSim.rho_sub_experiments, 
									  self.RhoEst.MeasSim.rho_sub_experiments), 1) - 1]

		#Create confidence interval
		conf_interval = (b_l, b_u)

		# Debugging
		# print("Lower and upper (adjusted) percentile bounds:")
		# print(adj_low_bound)
		# print(adj_upper_bound)

		return conf_interval

	####################################################################################################################
	#			Helper functions necessary to characterize the nature of the confidence interval 
	####################################################################################################################

	### Helper function for all intervals to get interval bounds based on alpha ###
	def __getPercBounds(self):
		"""
		Gives the percentile bounds of the confidence
		interval, relative to the "sided" parameter 
		given to the coverage class.
		"""

		#Initialize low and high bound vars
		low = None
		high = None

		if self.sided == "two_sided":
			low = (self.alpha)/2
			high = (1 - low)

		elif self.sided == "lower":
			low = 0
			high = 1 - self.alpha

		elif self.sided == "upper":
			low = self.alpha
			high = 1

		#Return the percent bounds
		return (low, high) 

	####################################################################################################################
	#			Helper functions necessary for running bias corrected and BC_a confidence intervals
	####################################################################################################################

	### Bias term z0 logic ###
	def __z_0(self, sample):
		"""
		Private helper function to calculate bias term for CI_BC_a and CI_BC. 
		Represents the bias or skew in the bootstrapped distribution and is used
		to try and adjust for it. 

		Args:
			sample:				Sample of rho_sub_i_hat_fid estimates

		Returns:
			z_0:				Bias term z0 (sometimes called "b" in literature)

		"""

		#Sum of the number of estimates less than the initial estimate for fidelity
		numerator = np.size([item for item in sample if item < self.RhoEst.rho_hat_fid])

		# Print numerator for debugging purposes (should not be greater than number of sub-experiments)
		# print("\nNumerator: \n%s\n"%numerator)

		#Need at least one to be greater than at minimum, otherwise this fails
		return norm.ppf(min(max(1,numerator),self.RhoEst.MeasSim.rho_sub_experiments - 1)/
						float(self.RhoEst.MeasSim.rho_sub_experiments))

	### Acceleration term acc logic ###
	def __acceleration(self, sample):
		"""
		Private helper function for CI_BC_a confidence interval method. 
		Determines the acceleration term of the bootstrapped distribution
		as a function of theta_dot and theta_i. Function taken from:

		"Efron, B., & Tibshirani, R. J. (1994). An introduction to the bootstrap. CRC press.""
		pg. 186

		Args:
			sample:				Sample of rho_sub_i_hat_fid estimates

		Returns:
			acc:				acc as a function of theta_dot and theta_i
		"""

		#Calculate theta_dot
		theta_dot = self.__theta_dot(sample)

		num_sum = 0
		den_sum = 0

		#Iterate through all theta_is and calculate numerator / denom
		for index in range(self.RhoEst.MeasSim.rho_sub_experiments):

			#Base represents the difference in theta_dot and theta_i
			base = (theta_dot - self.__theta_i(sample,index))

			# Numerator and denom exponentiate base differently
			num_sum += (base)**3
			den_sum += (base)**2

		# Calculate acceleration and return it
		acc = num_sum / (6*((den_sum)**(3/2)))

		return acc

	def __theta_dot(self, sample):
		"""
		Private helper function for CI_BC_a confidence interval method. 
		determines the average theta_i (described below) across the 
		entire sample.

		Args:
			sample:				Sample of rho_sub_i_hat_fid estimates

		Returns:
			theta_dot:			Average theta_i across the sample
		"""

		# Initialize a theta_sum
		theta_sum = 0

		# Iterate through the sub_experiment indices and sum up
		# The theta_i for each
		for index in range(self.RhoEst.MeasSim.rho_sub_experiments):
			theta_sum += self.__theta_i(sample,index)

		#Return the sum of theta_is divided by the number of sub experiments
		theta_dot = theta_sum / self.RhoEst.MeasSim.rho_sub_experiments

		return theta_dot

	def __theta_i(self, sample, index_to_remove):
		"""
		Private helper function for CI_BC_a confidence interval method. 
		Determines theta_i = S(x_(i)). S(x_(i)) is defined below, but the
		distribution x_(i) indicates the sample of rho estimates with the 
		ith element deleted. 

		Args:
			sample:				Sample of rho_sub_i_hat_fid estimates

		Returns:
			theta_i:			Average theta_i across the sample
		"""

		#Create x_(i) 
		newSamp = sample[:index_to_remove] + sample[index_to_remove + 1:]

		# Create theta_i	and return it. Theta i depends on the 
		# __s_x_BC sub function.
		theta_i = self.__s_x_BC_a(newSamp)

		return theta_i

	def __s_x_BC_a(self, sample):
		"""
		Private helper function to calculate S(x_(i)) for bias corrected 
		and accelerated method. S(x) is defined as the squared deviations
		from the mean of bootstrapped observations for the sample x. 

		Args:
			sample:				Sample of rho_sub_i_hat_fid estimates

		Returns:
			s_x_i:				s(x_(i)), or the deviations from the mean of 
								the bootstrapped sample observations. 
		"""

		#Get the mean of the sample, plus
		#initialize the sq_dif_sum
		sample_mean = np.mean(sample)
		sq_dif_sum = 0

		#For all items in the sample (minus the one left out)
		for index in range(self.RhoEst.MeasSim.rho_sub_experiments - 1):
			sq_dif_sum += (sample_mean - sample[index])**2

		#Calculate s_x_i and return the value
		s_x_i = (sq_dif_sum / self.RhoEst.MeasSim.rho_sub_experiments)

		return s_x_i
	

	