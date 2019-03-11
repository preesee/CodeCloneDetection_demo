import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np
import sys
from matplotlib import gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
def _has_git():
    try:
        subprocess.check_call(['git', '--version'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError):
        return False
    else:
        return True
def showAttention(input_code, input_txt, output_sentence, attention1, attention2, fig_filename):
    # Set up figure with colorbar
    fig = plt.figure()
    # f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)

    gs = gridspec.GridSpec(1, 2, width_ratios=[10, 21])
    ax1 = plt.subplot(gs[0])
    ax1.matshow(attention1, cmap='Blues')
    ax2 = plt.subplot(gs[1])
    mat2 = ax2.matshow(attention2, cmap='Blues')

    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(mat2, cax=cax, ticks=[0, 0.5, .99])
    cbar.ax.set_yticklabels(['0.0', '0.5', '1.0'])  # horizontal colorbar

    # Set up axes
    ax1.set_xticklabels([''] + input_code.split(' ') + ['<EOS>'], rotation=90)
    ax1.set_yticklabels([''] + output_sentence.split(' ')+ ['<EOS>'])
    # Show label at every tick
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(1))
    # Set up axes
    ax2.set_xticklabels([''] + input_txt.split(' ') + ['<EOS>'], rotation=90)
    # ax2.set_yticklabels([''] + output_sentence.split(' ') + ['<EOS>'])
    ax2.set_yticklabels([])
    # Show label at every tick
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(1))

    print("show..")
    plt.tight_layout()
    fig.savefig(fig_filename, bbox_inches='tight')
    # plt.show()

attn_path = 'vis.mtx'
input_code = "def subprocess subprocess OSError False 'git' '--version' subprocess True <EOS>"

# input_txt = 'x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x11 x12 x13 x14 x15 x16 x17 x18 x19 x20 x21'
input_txt = "def <unk> try subprocess check_call ['git' '--version'] stdout subprocess DEVNULL stderr subprocess DEVNULL except OSError subprocess CalledProcessError return <unk> return <EOS>"
output_sentence = "check if git is installed ."

attn_code = np.zeros((7, 10))
attn_code_lines = []
with open(attn_path, 'r') as file:
    for line in file.readlines():
        if line.startswith('attn_tree:'):
            attn_code_lines.append(line.strip('\n').split(':')[1])

attn_code_lines = attn_code_lines[-7:]
# print attn_code_lines
for i in range(7):
    attn_code[i] = np.fromstring(attn_code_lines[i], sep=' ')
# print "attn_code"
# print attn_code

# showAttention(input_code, output_sentence, attn_code, 'attn_code.pdf')
# sys.exit()

attn_txt = np.zeros((7, 21))
attn_txt_lines = []
with open(attn_path, 'r') as file:
    for line in file.readlines():
        if line.startswith('attn_txt:'):
            attn_txt_lines.append(line.strip('\n').split(':')[1])

attn_txt_lines = attn_txt_lines[-7:]
# print attn_txt_lines
for i in range(7):
    attn_txt[i] = np.fromstring(attn_txt_lines[i], sep=' ')
# print "attn_txt"
# print attn_txt

showAttention(input_code, input_txt, output_sentence, attn_code, attn_txt, 'visual_att_1.pdf')
