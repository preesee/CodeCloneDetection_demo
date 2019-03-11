import matplotlib.pyplot as plt
# plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np

def showAttention(input_sentence, output_sentence, attentions, fig_filename):
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions, cmap='bone') # Blues
    # fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sentence.split(' ') + ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_sentence.split(' ')+ ['<EOS>'])

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    print("show..")
    # fig.savefig(fig_filename, bbox_inches='tight')
    plt.show()

attn_path = 'vis.mtx'
input_code = 'c1 c2 c3 c4 c5 c6 c7 c8 c9 c10'
input_txt = 'x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x11 x12 x13 x14 x15 x16 x17 x18 x19 x20 x21'
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

showAttention(input_code, output_sentence, attn_code, 'attn_code.pdf')


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

showAttention(input_txt, output_sentence, attn_txt, 'attn_txt.pdf')
