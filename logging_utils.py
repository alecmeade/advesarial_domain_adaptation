import os 
import shutil
import re
import torch

TRAIN_PREFIX = "train"
EVAL_PREFIX = "eval"
ADAPT_PREFIX = "adapt"

class Logger():

    def __init__(self, log_dir, log_folder, delim=" | "):
        self.log_dir = os.path.join(log_dir, log_folder)
        self.delim = delim
        self.log = None

        if os.path.exists(self.log_dir):
            shutil.rmtree(self.log_dir)
        
        os.mkdir(self.log_dir)      

    def create_log(self, name):
        self.log = open(os.path.join(self.log_dir, ("%s.txt" % name)), "w")


    def write_log_line(self, line_vals):
        self.log.write(self.delim.join(line_vals))
        self.log.write("\n")


    def close_log(self):
        self.log.close()
        self.log = None


    def get_model_score(self, model_name):
        return float(model_name.split("_")[-2])


    def save_model(self, prefix, model, epoch, score, keep=3):
        model_name = ("%s_%d_%0.4f_.pt" % (prefix, epoch, score))

        models = []
        for f in os.listdir(self.log_dir):
            if f.startswith(prefix) and f.endswith('.pt'):
                score = self.get_model_score(f)
                models.append([f, score])
                models.sort(key=lambda x: x[1])

                if len(models) >= keep:
                    os.remove(os.path.join(self.log_dir, models[0][0]))
                    models.pop(0)

        torch.save(model, os.path.join(self.log_dir, model_name))


    def get_best_model(self, prefix):
        max_score = 0
        max_model_name = None
        for f in os.listdir(self.log_dir):
            if f.startswith(prefix) and f.endswith('.pt'):
                model_score = self.get_model_score(f)
                if model_score > max_score:
                    max_score = model_score
                    max_model_name = f
        
        return max_model_name

    def load_best_model(self, prefix):
        max_model_name = self.get_best_model(prefix)
        return torch.load(os.path.join(self.log_dir, max_model_name))