import os
import torch
import datetime
import numpy as np
import matplotlib.pyplot as plt


def train(net, normal_loader, abnormal_loader, optimizer, criterion):
    net.train()
    net.flag = "Train"
    ninput, nlabel = next(normal_loader)
    ainput, alabel = next(abnormal_loader)
    _data = torch.cat((ninput, ainput), 0)
    _label = torch.cat((nlabel, alabel), 0)
    _data = _data.cuda()
    _label = _label.cuda()
    predict = net(_data)
    cost, loss = criterion(predict, _label)
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    return loss

def plot_losses_onepicture(loss_history, save_path=None):
    plt.figure(figsize=(15, 10))
    for key in loss_history.keys():
        plt.plot(loss_history[key], label=key)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()


def plot_losses(loss_history, num_steps_per_epoch, save_path=None):
    num_plots = len(loss_history.keys())
    num_columns = 2
    num_rows = (num_plots + 1) // num_columns

    fig, axes = plt.subplots(num_rows, num_columns, figsize=(15, num_rows * 5))
    x_axis = np.arange(1, len(loss_history[list(loss_history.keys())[0]]) + 1) / num_steps_per_epoch

    for idx, key in enumerate(loss_history.keys()):
        row = idx // num_columns
        col = idx % num_columns
        axes[row, col].plot(x_axis, loss_history[key], label=key)
        axes[row, col].set_xlabel('Epoch')
        axes[row, col].set_ylabel('Loss')
        axes[row, col].legend()
        axes[row, col].grid()

    # Remove unused subplots
    for idx in range(num_plots, num_rows * num_columns):
        row = idx // num_columns
        col = idx % num_columns
        fig.delaxes(axes[row, col])

    fig.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)
    plt.show()


def save_experiment_results(experiment_name, run_number, result1, result2, result3, filename="experiment_results.txt"):
    # Check if the file exists
    file_exists = os.path.isfile(filename)
    experiment_exists = False
    line_to_update = -1

    if file_exists:
        with open(filename, "r") as file:
            lines = file.readlines()
            in_experiment_block = False
            for idx, line in enumerate(lines):
                if line.strip() == f"Experiment_name: {experiment_name}":
                    experiment_exists = True
                    in_experiment_block = True
                elif "=" * 80 in line:
                    in_experiment_block = False
                elif in_experiment_block and f"{run_number}\t" in line:
                    line_to_update = idx
                    break

    # If the line_to_update is not -1, update the line with the new results
    if line_to_update != -1:
        now = datetime.datetime.now().strftime("%Y-%m-%d")
        new_line = f"{run_number}\t{now}\t{result1}\t{result2}\t{result3}\n"
        lines[line_to_update] = new_line

        with open(filename, "w") as file:
            file.writelines(lines)
    else:
        # Open the file in append mode
        with open(filename, "a") as file:
            # If the experiment name is not in the file, add a separator, the name and a header
            if not experiment_exists:
                separator = "\n" + "=" * 80 + "\n"
                header = f"Experiment_name: {experiment_name}\n"
                title = "Run_Number\tDate\tauc\tap\tac\n"
                file.write(separator + header + title)

            # Add the experiment results
            now = datetime.datetime.now().strftime("%Y-%m-%d")
            line = f"{run_number}\t{now}\t{result1}\t{result2}\t{result3}\n"
            file.write(line)
