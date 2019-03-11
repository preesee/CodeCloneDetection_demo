import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import matplotlib.gridspec as gridspec
import sys
log_file = "/home/wanyao/log/code_summarization/log.a2c-train_10_30_10_code_1_g3"

# def example_plot(ax, fontsize=12):
#     ax.plot([1, 2])
#     ax.locator_params(nbins=3)
#     ax.set_xlabel('x-label', fontsize=fontsize)
#     ax.set_ylabel('y-label', fontsize=fontsize)
#     ax.set_title('Title', fontsize=fontsize)

with open(log_file, 'r') as file:
    fig = plt.figure()
    iteration, loss, reward = [], [], []
    for line in file.readlines():
        if line.startswith("iteration"):
            iteration.append(int(line.strip('\n').strip().split(",")[0].split(":")[1]))
            # print line.strip('\n').strip().split(",")[1].split(":")[1].strip()
            if line.strip('\n').strip().split(",")[1].split(":")[0].strip() == 'loss':
                loss.append(float(line.strip('\n').strip().split(",")[1].split(":")[1]))
            elif line.strip('\n').strip().split(",")[1].split(":")[0].strip() == 'reward':
                reward.append(float(line.strip('\n').strip().split(",")[1].split(":")[1]))
    print(len(iteration))
    print (len(loss))
    print (len(reward))
    # sys.exit()
    gs1 = gridspec.GridSpec(1, 2)
    ax1 = fig.add_subplot(gs1[0])
    ax2 = fig.add_subplot(gs1[1])

    fontsize = 12
    loss = loss[0:7803] + [loss[7802]]*(16473-7803)
    ax1.plot(iteration[0:len(loss)], loss)
    ax1.locator_params(nbins=3)
    ax1.set_xlabel('iteration', fontsize=fontsize)
    ax1.set_ylabel('cross entropy loss', fontsize=fontsize)
    ax1.text(4000, 7000, 'pretrain', fontsize=15)

    # ax1.set_title('Title', fontsize=fontsize)
    ax1.spines['right'].set_color('none')
    print (reward[0:10])
    ax2.plot(iteration[-len(reward):], reward)
    ax2.locator_params(nbins=3)
    ax2.set_xlabel('iteration', fontsize=fontsize)
    ax2.set_ylabel('reward', fontsize=fontsize)
    ax2.text(20000, 34, 'reinforce', fontsize=15)

    # ax2.set_title('Title', fontsize=fontsize)

    gs1.tight_layout(fig, w_pad=-5.1, rect=[0, 0, 1, 1])

    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # plt.xlim(-1,7803)
    # ax.plot(x, y, label='y = numbers')

    fig.savefig('/home/wanyao/www/Dropbox/ghproj-py36/code_summarization/lib/visual/iteration.pdf')
