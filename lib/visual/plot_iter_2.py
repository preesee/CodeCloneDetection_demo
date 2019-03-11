import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import matplotlib.gridspec as gridspec
import sys
log_file = "log.a2c-train_10_30_10_hybrid_1_g0"

# def example_plot(ax, fontsize=12):
#     ax.plot([1, 2])
#     ax.locator_params(nbins=3)
#     ax.set_xlabel('x-label', fontsize=fontsize)
#     ax.set_ylabel('y-label', fontsize=fontsize)
#     ax.set_title('Title', fontsize=fontsize)

with open(log_file, 'r') as file:
    iteration, loss, reward, perplexity = [], [], [], []
    for line in file.readlines():
        if line.startswith("Epoch"):
            if 'perplexity' in line:
                perplexity.append(float(line.strip('\n').split(';')[1].split(':')[1].strip()))
                iteration.append(int(line.strip('\n').split(';')[0].split(',')[1].split('/')[0].strip()))
            if 'reward' in line:
                reward.append(float(line.strip('\n').split(';')[1].split(':')[1].strip()))
                iteration.append(int(line.strip('\n').split(';')[0].split(',')[1].split('/')[0].strip()))

# with open(log_file, 'r') as file:
#     fig = plt.figure()
#     iteration, loss, reward = [], [], []
#     for line in file.readlines():
#         if line.startswith("iteration"):
#             iteration.append(int(line.strip('\n').strip().split(",")[0].split(":")[1]))
#             # print line.strip('\n').strip().split(",")[1].split(":")[1].strip()
#             if line.strip('\n').strip().split(",")[1].split(":")[0].strip() == 'loss':
#                 loss.append(float(line.strip('\n').strip().split(",")[1].split(":")[1]))
#             elif line.strip('\n').strip().split(",")[1].split(":")[0].strip() == 'reward':
#                 reward.append(float(line.strip('\n').strip().split(",")[1].split(":")[1]))

    fig = plt.figure(figsize=(9,  3), dpi=72.0)

    gs1 = gridspec.GridSpec(1, 2)
    ax1 = fig.add_subplot(gs1[0])
    # ax1.set_aspect(0.06)
    ax2 = fig.add_subplot(gs1[1])
    # ax2.set_aspect(0.2)

    fontsize = 16
    # loss = loss[0:7803] + [loss[7802]]*(16473-7803)
    ax1.plot(range(len(perplexity)), perplexity, linewidth=2.0)
    # ax1.locator_params(nbins=3)
    ax1.set_xlabel('iteration', fontsize=fontsize)
    ax1.set_ylabel('perplexity', fontsize=fontsize)
    # ax1.text(4000, 7000, 'pretrain', fontsize=15)

    # ax1.set_title('Title', fontsize=fontsize)
    # ax1.spines['right'].set_color('none')
    print(reward[0:10])

    ax2.plot(range(len(reward)), reward, linewidth=2.0)
    # ax2.locator_params(nbins=3)
    ax2.set_xlabel('iteration', fontsize=fontsize)
    ax2.set_ylabel('reward', fontsize=fontsize)
    ax2.set_ylim((0,40))
    # ax2.text(20000, 34, 'reinforce', fontsize=15)

    # ax2.set_title('Title', fontsize=fontsize)

    plt.tight_layout()
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # plt.xlim(-1,7803)
    # ax.plot(x, y, label='y = numbers')
    # plt.show()
    # fig.set_size_inches(20, 10)

    fig.savefig('iteration.pdf')
