###########################################################################
#
# Logger file for collecting execution metadata and results 
# 
# Author: Sam Showalter
# Date: June 5, 2018
#
###########################################################################

###########################################################################
# Module and library imports
###########################################################################

import datetime as dt 
import pandas as pd 
import os

###########################################################################
# Class and Constructor
###########################################################################

class Log():
	"""
	Log object corresponds to a test in two ways. A Master_Log can be thought of
	as a snapshot of an entire execution, from start to finish. All data collected 
	is stored at a summary level in this log. One Master_Log row corresponds to a 
	single test, and naturally more than one test can be stored in a master log.

	Additionally, Boot_Logs store the granular information found when bootstrapping. 
	Each boot log file corresponds to a SINGLE test. Each Master_Log row will include the file
	name of the corresponding Boot_Log file. All boot_log files are stored in folders relative
	to the master_log which housing the execution summary. One folder corresponds to one Master_Log

	Attributes:

			### Log Names ###

		master_log_name:					Name provided for the master log
		boot_log_name:						Name provided for the boot log
		verbose_log_name:					Name provided for verbose MLE execution data log
		directory:							Directory where all log data will be stored 
											(don't forget to add escape characters when necessary)

			### Logs ####

		master_log:							Dataframe corresponding to master_log data (provided)
		boot_log:							Dataframe corresponding to boot_log data (provided)
		verbose_log:						Dataframe corresponding to verbose_log data (MLE execution data)
	"""

	def __init__(self, master_log_name, 
					   boot_log_name, 
					   verbose_log_name,
					   master_col_names = 
									   		#Execution Metadata
										    [["Execution_Date",
											"Execution_Time",
											"Execution_Duration_Sec",
											
											#Information from user inputs for MeasSim
											"Dimensions",
											"Tolerance",
											"Random_Seed",
											"Rho_Goal",
											"Rho_Goal_Purity",
											"Noise_Perc",
											"Rho_True",
											"Rho_True_Fidelity",
											"Rho_True_Purity",
											"Mean_Boot_Rho_Purity",
											"Rho_True_Log_Lik",
											"Mean_Boot_Rho_Log_Lik",
											"Initial_Rho_Hat_Trials",
											"Rho_Sub_Experiments",
											"Rho_Sub_Trials",

											#Output from MeasSim execution (first pass)
											"Rho_Hat_Probs",

											#Input to RhoEst and Solver classes
											"Solver_Method",
											"Stopping_Criteria",
											"Epsilon_Corrections",

											#Output from RhoEst and Solver execution
											"RrhoR_iterations",
											"Rho_Hat",
											"Rho_Hat_Fidelity",
											"Rho_Hat_Log_Lik",
											"Rho_Hat_Purity",
											"Rho_Hat_Pos_Corr",
											"Rho_Hat_MLE_Iter_Bound",
											"Rho_Hat_MLE_Timeout",

											#Bootstrapped output Metadata
											"Boot_Rho_Hat_Pos_Corr",

											#Coverage interval input
											"Confidence",
											"Conf_Sided",
											"Conf_Method",

											#Coverage interval output FIDELITY (low, high)

											#Normal distribution
											"CI_Fid_Norm_Low",
											"CI_Fid_Norm_High",

											#Basic method
											"CI_Fid_Basic_Low",
											"CI_Fid_Basic_High",

											#Percentile method
											"CI_Fid_Percentile_Low",
											"CI_Fid_Percentile_High",

											#Bias Corrected method
											"CI_Fid_BC_Low",
											"CI_Fid_BC_High",

											#Bias Corrected and accelerated method
											"CI_Fid_BC_a_Low",
											"CI_Fid_BC_a_High",


											#Coverage interval output PURITY (low, high)

											#Normal distribution
											"CI_Purity_Norm_Low",
											"CI_Purity_Norm_High",

											#Basic method
											"CI_Purity_Basic_Low",
											"CI_Purity_Basic_High",

											#Percentile method
											"CI_Purity_Percentile_Low",
											"CI_Purity_Percentile_High",

											#Bias Corrected method
											"CI_Purity_BC_Low",
											"CI_Purity_BC_High",

											#Bias Corrected and accelerated method
											"CI_Purity_BC_a_Low",
											"CI_Purity_BC_a_High",

											#Results log filename
											"Boot_Log_Filename"]],

					   boot_col_names = 	#Bootstrap column names, metadata
					   						[["Boot_Execution_Date",
											  "Boot_Execution_Time",
											  "Boot_Duration_Sec",

											  # Bootstrap output data
											  "Boot_Rho_Prob",
											  "Boot_RrhoR_Iter",
											  "Boot_epsilon_corrections",
											  "Boot_Rho_Hat",
											  "Boot_Rho_Fid",
											  "Boot_Rho_Log_Lik",
											  "Boot_Rho_Purity",
											  "Boot_Rho_Pos_Corr",
											  "Boot_MLE_Timeout",
											  "Verbose_Log_Filename"]],

						verbose_col_names = [["Rho",
											  "R",
											  "Epsilon",
											  "Upper_Bound",
											  "Log_Likelihood"]],

					   directory = "C:\\Users\\srs9\\Documents\\State_Tomo_Coverage_Data\\"):

		"""
		Constructor for Log object. Creates dataframe logs and sets the correct directory

		Args:

				### Log Names ###

			master_log_name:					Name provided for the master log
			boot_log_name:						Name provided for the boot log
			master_col_names:					Column names for the master log
			boot_col_names:						Column names for the boot log
			directory:							Directory where all log data will be stored 
												(don't forget to add escape characters when necessary)

		"""

		#Set the directory to store all data output
		self.directory = directory

		#Log names
		self.master_log_name = master_log_name + str(dt.datetime.now().strftime("_%Y-%m-%d_%H.%M.%S"))
		self.boot_log_name = boot_log_name + str(dt.datetime.now().strftime("_%H.%M.%S.%f")[:-3])
		self.verbose_log_name = verbose_log_name + str(dt.datetime.now().strftime("_%H.%M.%S.%f")[:-3])

		#Two DataFrame logs for performance and data collection
		self.master_log = pd.DataFrame(columns = master_col_names)
		self.boot_log = pd.DataFrame(columns = boot_col_names)
		self.verbose_log = pd.DataFrame(columns = verbose_col_names)


	###########################################################################
	# Public methods that facilitate execution of logger
	###########################################################################

	def addMasterLogRecord(self, test):
		"""
		Adds a record to the master log. (One record = one test)

		Args:

			test:					test (QSTSim) object containing all data from test
			
		"""
		new_record_df = pd.DataFrame(
								      [[
								       #Test Execution Information
								       test.execution_date_start,
								       test.execution_time_start,
								       test.duration_sec,

								       #Information from User Inputs for MeasSim (no bootstrapping yet)
								       test.MeasSim.dims,									#Dimension in Hilbert space
								       test.MeasSim.tol,									#Tolerance level for floating point errors
								       test.MeasSim.seed,									#Random seed for control of number generation
								       test.MeasSim.rho_goal,								#Rho goal
								       test.MeasSim.rho_goal_purity,						#Rho_goal purity
								       test.MeasSim.noise_perc,								#Noise percentage
								       test.MeasSim.rho_true,								#Rho true
								       test.MeasSim.rho_true_fidelity,						#Rho true fidelity
								       test.MeasSim.rho_true_purity,						#Rho true purity
								       test.RhoEst.mean_boot_purity,						#Mean boot rho purity 
								       test.MeasSim.rho_true_log_lik,						#Rho true log likelihood
								       test.RhoEst.mean_boot_log_lik,						#Mean bootstrapped log likelihood
								       test.MeasSim.rho_hat_trials,							#Meas trials for initial Rho_hat
								       test.MeasSim.rho_sub_experiments,					#Number of bootstrapped samples to be made from rho_hat
								       test.MeasSim.rho_sub_trials,							#Meas trials for bootstrapped rho_hat
								      
								       #Output from MeasSim execution
								       test.MeasSim.rho_hat_probs,							#Rho_hat measurement probabilities
								       
								       #Input to RhoEst and Solver classes
								       test.RhoEst.Solver.solver_method,					#Solver method
								       test.RhoEst.Solver.stopping_crit,					#Stopping criteria bound (MLE only!)
								       test.RhoEst.Solver.epsilon_corrections,				#Number of epsilon corrections
								       
								       #Output from RhoEst and Solver execution
								       test.RhoEst.Solver.RrhoR_iters,						#Number of RrhoR iterations (MLE only!)	
								       test.RhoEst.rho_hat,									#Rho_hat initial estimate, made from rho_true
								       test.RhoEst.rho_hat_fid,								#Rho_hat fidelity relative to rho_goal
								       test.RhoEst.rho_hat_log_lik,							#Log likelihood of rho-hat
								       test.RhoEst.rho_hat_purity,							#Purity of rho_hat calculations
								       test.RhoEst.MeasSim.rho_hat_pos_corr,				#Rho_hat positivity correction boolean
								       test.RhoEst.Solver.iter_bound,					 	#Number of MLE iterations before timeout
								       test.RhoEst.Solver.MLE_timeout,						#MLE_timeout for initial rho_hat

								       #Bootstrapped output metadata
								       test.MeasSim.boot_rho_hat_pos_corr_count,			#Bootstrap rho positivity correction count

								       #Coverage interval input
								       test.Coverage.confidence,							#Coverage confidence
								       test.Coverage.sided,									#Type of sided CI
								       test.Coverage.method,								#Method of CI implemented

								       #Coverage interval output for FIDELITY (low, high)
								       test.Coverage.intervals["fidelity"]["normal"][0],			#Low bound
								       test.Coverage.intervals["fidelity"]["normal"][1],			#Upper bound

								       test.Coverage.intervals["fidelity"]["basic"][0],				#Low bound
								       test.Coverage.intervals["fidelity"]["basic"][1],				#Upper bound

								       test.Coverage.intervals["fidelity"]["percentile"][0],		#Low bound
								       test.Coverage.intervals["fidelity"]["percentile"][1],		#Upper bound

								       test.Coverage.intervals["fidelity"]["bias_corrected"][0],	#Low bound
								       test.Coverage.intervals["fidelity"]["bias_corrected"][1],	#Upper bound

								       test.Coverage.intervals["fidelity"]["BC_a"][0],				#Low bound
								       test.Coverage.intervals["fidelity"]["BC_a"][1],				#Upper bound

								       #Coverage interval output for PURITY (low, high)
								       test.Coverage.intervals["purity"]["normal"][0],				#Low bound
								       test.Coverage.intervals["purity"]["normal"][1],				#Upper bound

								       test.Coverage.intervals["purity"]["basic"][0],				#Low bound
								       test.Coverage.intervals["purity"]["basic"][1],				#Upper bound

								       test.Coverage.intervals["purity"]["percentile"][0],			#Low bound
								       test.Coverage.intervals["purity"]["percentile"][1],			#Upper bound

								       test.Coverage.intervals["purity"]["bias_corrected"][0],		#Low bound
								       test.Coverage.intervals["purity"]["bias_corrected"][1],		#Upper bound

								       test.Coverage.intervals["purity"]["BC_a"][0],				#Low bound
								       test.Coverage.intervals["purity"]["BC_a"][1],				#Upper bound

								       #Relevant bootstrap log file
								       test.RhoEst.BootLogFilename]],			

									   #Add the Master Log Column Names
									   columns = self.master_log.columns)

		#Add record to the existing dataframe
		self.master_log = pd.concat([self.master_log, new_record_df], axis = 0)
		self.master_log.reset_index(drop = True, inplace = True)

	def addBootRecord(self, RhoEst):
		"""
		Adds a record to the boot log. (One record = one test)

		Args:

			RhoEst:					RhoEstimator object, where all of the bootstrapping occurs.
			
		"""
		new_metadata_df = pd.DataFrame(
								      [[
								      	#Test Execution Information
								      	RhoEst.Solver.start_date,					#Current date
								       	RhoEst.Solver.start_time,					#Current time
								       	RhoEst.Solver.duration_sec,					#Model duration in seconds

								       	#Test output information
								       	RhoEst.boot_rho_prob,						#Bootstrapped probability measurements (freq)
								       	RhoEst.Solver.boot_RrhoR_iter,				#Bootstrapped RrhoR iterations
								       	RhoEst.Solver.epsilon_corrections,			#Number of epsilon corrections for bootstrapping
								       	RhoEst.boot_rho_hat, 						#Bootstrapped rho hat
								       	RhoEst.boot_rho_fid,						#Bootstrapped fidelity	
								       	RhoEst.boot_rho_log_lik,					#Log likelihood of Bootstrapped sample
								       	RhoEst.boot_rho_purity,						#Purity of the bootstrapped sample
								       	RhoEst.MeasSim.boot_rho_hat_pos_corr,		#Bootstrapped positivity correction
								       	RhoEst.Solver.MLE_timeout,					#MLE timeout for bootstrapping (MLE only)
								       	RhoEst.Solver.verbose_log_filename]],		#Verbose log filename

									   #Add the Collection Log Column Names
									   columns = self.boot_log.columns)

		#Add bootstrapped record to the boot_log
		self.boot_log = pd.concat([self.boot_log ,new_metadata_df], axis = 0)
		self.boot_log.reset_index(drop = True, inplace = True)

	#Add record for MLE execution
	def addVerboseRecord(self, MLE):
		"""
		Adds a record to the verbose log for MLE execution. (One record = one optimization iteration)

		Args:

			MLE:					MLE solver object
			
		"""

		verbRecord = pd.DataFrame([[#MLE execution information
									MLE.current_rho,
									MLE.R,
									MLE.epsilon,
									MLE.upper_bound,
									MLE.log_likelihood]],

					 columns = self.verbose_log.columns)

		#Add verbose record to the verbose_log
		self.verbose_log = pd.concat([self.verbose_log ,verbRecord], axis = 0)
		self.verbose_log.reset_index(drop = True, inplace = True)



	def saveMasterLog(self):
		"""
		Saves the master log to a .csv file (can be changed to .txt and others) in the appropriate directory
			
		"""
		#Change working directory for Master Logs
		os.chdir(self.directory)

		#Save the master_log
		self.master_log.to_csv(self.master_log_name + ".csv", sep = ",")

	def saveBootLog(self):
		"""
		Saves the boot log to a .csv file (can be changed to .txt and others) in the appropriate directory
			
		"""

		#Change working directory for Result Logs
		os.chdir(self.directory)

		#Check to see if a master_log folder has been created
		if not os.path.exists(self.directory + self.master_log_name):
			os.makedirs(self.master_log_name)

		#Navigate to the appropriate directory
		os.chdir(self.directory + self.master_log_name)

		#Save the boot log
		self.boot_log.to_csv(self.boot_log_name + ".csv", sep = ",")
	
	def saveVerboseLog(self):
		"""
		Saves the verbose log to a .csv file (can be changed to .txt and others) in the appropriate directory
			
		"""

		#Change working directory for Result Logs
		os.chdir(self.directory)

		#Check to see if a master_log and boot log folder have been created
		if not os.path.exists(self.directory + self.master_log_name + "\\" + self.boot_log_name):
			os.makedirs(self.directory + "\\" + self.master_log_name + "\\" + self.boot_log_name)
			
		#Change the directory to within the boot_log
		os.chdir(self.directory + "\\" + self.master_log_name + "\\" + self.boot_log_name)

		#Save the boot log
		self.verbose_log.to_csv(self.verbose_log_name + ".csv", sep = ",")
