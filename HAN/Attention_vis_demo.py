# coding: utf-8
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class CharVal(object):
    def __init__(self, char, val):
        char=   char#list(map(sequence_to_text, char))
        self.char = char#
        self.val = val

    def __str__(self):
        return self.char
def highlight(word, attn):
    html_color = '#%02X%02X%02X' % ( int(255*(1 - attn)), int(255*(1 - attn)),255)
    return '<span style="background-color: {}">{}</span>'.format(html_color, word)

def mk_html(seq, attns):
    html = ""
    for ix, attn in zip(seq, attns):
        html += ' ' + highlight(
            ix,
            attn
        )
    return html + "<br>"

from IPython.display import HTML, display
batch_size = 1
seqs = ["这", "是", "一个", "测试", "样例", "而已"]
attns = [0.01, 0.19, 0.12, 0.7, 0.2, 0.1]
#char_vals = [CharVal(c, v) for c, v in zip(seqs, attns)]
import pandas as pd
#df = pd.DataFrame(char_vals).transpose()

df = pd.DataFrame({"Cuenta":  [0.01, 0.19, 0.12, 0.7, 0.2, 0.1]},
                  index=["this", "is", "a", "test", "sample", "only"])
sns.heatmap(df, cmap="YlGnBu")
sns.heatmap(df, cmap="Blues")
sns.heatmap(df, cmap="BuPu")
sns.heatmap(df, cmap="Greens")
plt.show()
for i in range(batch_size):
    text = mk_html(seqs[i], attns[i])
    #display(HTML(text))
    with open(r'c:\tmp\html.html', 'w') as html:
        html.write(text)
