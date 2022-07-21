import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import cv2


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def select_evenly_spaced_elements(num_elements, sequence_length):
    return [i * sequence_length // num_elements + sequence_length // (2 * num_elements) for i in range(num_elements)]



def plot_grad_flow(named_parameters):
    '''
    RETURNING EARLY
    Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    return
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            # remove the first part (module name) and last part ("weight") of the module name
            n = '.'.join(n.split('.')[1:-1])
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.5, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.5, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    plt.tight_layout()
    plt.savefig('/tmp/gradient_flow.pdf')
    plt.close()

def plot_grad_flow_bars(named_parameters, lr=1):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''

    # named_parameters_copy = named_parameters.copy().cpu()

    figsize = (10,10)
    fig, ax = plt.subplots(figsize=figsize)
    ave_grads = []
    max_grads = []
    min_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and p.grad is not None:
            layers.append(n)
            ave_grads.append(lr*p.grad.abs().mean().cpu())
            max_grads.append(lr*p.grad.abs().max().cpu())
            min_grads.append(lr*p.grad.abs().min().cpu())

    ax.bar(3*np.arange(len(max_grads)), max_grads, lw=2, color="r")
    ax.bar(3*np.arange(len(max_grads)), ave_grads, lw=2, color="m")
    ax.bar(3*np.arange(len(max_grads)), min_grads, lw=2, color="b")

    ax.set_xticks(range(0, 3*len(ave_grads), 3))
    labels = ax.set_xticklabels(layers)
    for l in labels:
        l.update({"rotation": "vertical"})

    ax.set_xlim(left=0, right=3*len(ave_grads))
    ax.set_ylim(bottom=1e-7*lr, top=1e2*lr)
    ax.set_yscale("log")# zoom in on the lower gradient regions
    ax.set_xlabel("Layers")
    ax.set_ylabel("average gradient")
    ax.set_title("Gradient flow")
    ax.grid(True)
    ax.legend([Line2D([0], [0], color="r", lw=4),
               Line2D([0], [0], color="m", lw=4),
               Line2D([0], [0], color="b", lw=4)], ['max-gradient', 'mean-gradient', 'min-gradient'])

    fig.tight_layout()
    return fig
