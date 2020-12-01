import argparse 
import csv
import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sns




def plot_charts(args):
    fpath = os.path.join(args.adda_path, 
                        ("adapt" + "_" + args.source_dataset + "_" + args.target_dataset + ".txt"))
    header = []
    vals = []
    with open(fpath, newline='') as f:
        reader = csv.reader(f, delimiter="|")
        for i, row in enumerate(reader):
            if i == 0:
                header = [c.strip() for c in row]
            else:
                vals.append([float(v.strip()) for v in row])

    adda_df = pd.DataFrame(vals, columns=header)

    fpath = os.path.join(args.wdgrl_path, 
                        ("train" + "_" + args.source_dataset + "_" + args.target_dataset + ".txt"))
    header = []
    vals = []
    with open(fpath, newline='') as f:
        reader = csv.reader(f, delimiter="|")
        for i, row in enumerate(reader):
            if i == 0:
                header = [c.strip() for c in row]
            else:
                vals.append([float(v.strip()) for v in row])


    wdgrl_df = pd.DataFrame(vals, columns=header)

    epochs = 300
    plt.figure()
    plt.title("Source: " + args.source_dataset + " | Target: " + args.target_dataset)
    plt.plot([0, 0] + list(adda_df.epoch[:300]), [0, 0] + list(adda_df.target_val_acc[:300]), label="ADDA")
    plt.plot([0, 0] + list(wdgrl_df.epoch[:300]), [0, 0] + list(wdgrl_df.target_val_acc[:300]), label="WDGRL")
    plt.ylim(0, 1)
    plt.ylabel("Validation Accuracy")
    plt.xlabel("Epoch")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--adda_path', type=str, default = "adda_logs/USPS_7000_2000_2")
    arg_parser.add_argument('--wdgrl_path', type=str, default = "wdgrl_logs/USPS_7000_2000_0")
    arg_parser.add_argument('--source_dataset', type=str, default = "USPS")
    arg_parser.add_argument('--target_dataset', type=str, default = "SVHN")
    args = arg_parser.parse_args()
    plot_charts(args)