import argparse
import os

import torch
import matplotlib.pyplot as plt
from matplotlib import cm


def plot_line(l, title=""):
    plt.plot(range(len(l)), l)
    plt.title(title)
    plt.yscale("log")
    plt.show()

def surface_plot(x):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    # Plot the surface.
    surf = ax.plot_surface(range(len(x)), range(len(x[0])), x, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()

def main(args):
    print("=> loading checkpoint '{}'".format(args.pretrained))
    epochs = [
        "0001", "0005", "0010", "0015", "0020", "0025", "0030", "0035",
        "0040", "0045", "0050", "0055", "0060", "0065", "0070", "0075",
        "0080", "0085", "0090", "0095", "0100"]
    w_1s = []
    w_2s = []
    b_2s = []
    figure_path = os.path.join(args.pretrained, "figures")
    for epoch in epochs:
        checkpoint_path = os.path.join(args.pretrained, f"checkpoint_{epoch}.pt")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        w_1s.append(checkpoint['state_dict']['module.predictor.predictor_512.0.weight'])
        w_2s.append(checkpoint['state_dict']['module.predictor.predictor_512.1.weight'])
        b_2s.append(checkpoint['state_dict']['module.predictor.predictor_512.1.bias'])

    w_s = []
    for w_1, w_2 in zip(w_1s, w_2s):
        w_s.append(w_2 @ w_1)

    ranks = [torch.matrix_rank(w) for w in w_s]
    singular_values = [torch.linalg.svd(w).S for w in w_s]
    min_Svs = [S.min() for S in singular_values]
    mean_Svs = [S.mean() for S in singular_values]
    max_Svs = [S.max() for S in singular_values]
    plot_line(min_Svs, "Min of singular values")
    plot_line(mean_Svs, "Mean of singular values")
    plot_line(max_Svs, "Max of singular values")
    # surface_plot(singular_values)
    pass
    for w, e in zip(w_s, epochs):
        plt.imshow(w, interpolation='none', aspect='auto')
        plt.title(f"Predhead matrix at {e} epoch")
        plt.savefig(os.path.join(figure_path, f"pred_head_matrix{e}"))
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--pretrained', type=str,
                        default="/storage/simsiam/logs/original_nobnnorelupredhead_384bs_512",
                        help='path to simsiam pretrained checkpoint')
    args = parser.parse_args()
    main(args)