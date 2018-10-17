class deRNN:
    def __init__(self):
        self.res = []  # the predicted value of yt
        pre_input = []  # xt that is combined by yt and ut
        ready_input = []  # zt computed by xt and add_input_layer
        internal_output = []  # ht computed by add_cell
        final_output = []  # P(t+1) computed by add_output_layer