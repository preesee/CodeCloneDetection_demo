# https://stackoverflow.com/questions/53867351/how-to-visualize-attention-weights
class CharVal(object):
    def __init__(self, char, val):
        self.char = char
        self.val = val

    def __str__(self):
        return self.char

def rgb_to_hex(rgb):
    return '#%02x%02x%02x' % rgb
def color_charvals(s):
    r = 255-int(s.val*255)
    color = rgb_to_hex((255, r, r))
    return 'background-color: %s' % color

# if you are using batches the outputs will be in batches
# get exact attentions of chars
MODEL_SAVING_DIR="C:\\work\\current_codeClone\\"
model_save_name='model.save'
model_wight_name=MODEL_SAVING_DIR+model_save_name+'.h5'
model = load_model(model_wight_name)
model.summary()
model = Model(inputs=model.input,
              outputs=[model.output, model.get_layer('attention_vec').output])

ouputs = model.predict(encoded_input_text)
model_outputs = outputs[0]
attention_outputs = outputs[1]


an_attention_output = attention_outputs[0][-len(encoded_input_text):]

# before the prediction i supposed you tokenized text
# you need to match each char and attention
char_vals = [CharVal(c, v) for c, v in zip(tokenized_text, attention_output)]
import pandas as pd
char_df = pd.DataFrame(char_vals).transpose()
# apply coloring values
char_df = char_df.style.applymap(color_charvals)
char_df


model = load_model("./saved_model.h5")
model.summary()
model = Model(inputs=model.input,
              outputs=[model.output, model.get_layer('attention_vec').output])

ouputs = model.predict(encoded_input_text)
model_outputs = outputs[0]
attention_outputs = outputs[1]