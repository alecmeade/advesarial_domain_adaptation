
import os 
import shutil
import re

LOG_DIR = 'logs/'


class Logger():

	def __init__(self, log_name, args, header_cols, delim="|"):
		self.log_name = log_name
		self.args = args
		self.log_dir = os.path.join(LOG_DIR, log_name)
		self.delim = delim
		if os.path.exists(self.log_dir):
			shutil.rmtree(self.log_dir)
		
		os.mkdir(self.log_dir)
		with open(os.path.join(self.log_dir, "params.txt"), "w") as f:
			f.write("PARAMS:\n")
			for name, value in vars(self.args).items():
				f.write("%s : %s\n" % (name, str(value)))

		self.log = open(os.path.join(self.log_dir, "log.txt"), "w")
		self.log.write(self.delim.join(header_cols))
		self.log.write("\n")

	def log_metrics_line(self, metrics):
		metrics_str = self.delim.join(metrics)
		self.log.write(metrics_str)
		self.log.write("\n")


	def close_log(self):
		self.log.close()

