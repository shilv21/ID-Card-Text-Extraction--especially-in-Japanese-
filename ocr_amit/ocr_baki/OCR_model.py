import cv2
import itertools
import numpy as np
import tensorflow as tf
import keras.backend
from tcn import TCN
from keras.layers import Input
from keras.layers import TimeDistributed, Bidirectional
from keras.layers import Dense, Permute, Flatten, CuDNNGRU, GRU, Lambda
from keras.models import Model
from keras.optimizers import SGD, Adam
from keras.backend import ctc_decode
from .backbone import dropout_vgg_h1, dropout_vgg
import json

keras.backend.set_image_dim_ordering('tf')

# graph = tf.get_default_graph()


class AlpsOcrModel:
    def __init__(self):
        self.label_text = None
        self.model = None
        self.model_json = None
        self.model_weight = None
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.session = tf.Session(graph=self.graph)

    def _ctc_lambda_func(self, args):
        y_pred, labels, input_length, label_length = args
        y_pred = y_pred[:, 2:, :]

        return keras.backend.ctc_batch_cost(labels, y_pred, input_length, label_length)

    def _build_basemodel(self, height, n_classes):
        rnnunit = 256
        inputs = Input(shape=(height, None, 1), name='the_input')

        # 1. CNN Feature Extractor
        m = dropout_vgg_h1(inputs)

        # 2. Bidirectional LSTM
        m = Permute((2, 1, 3), name='permute')(m)
        m = TimeDistributed(Flatten(), name='timedistrib')(m)
        m = TCN(nb_filters=256, kernel_size=3, nb_stacks=2, dilations=[1, 2, 4, 8, 16, 32, 64, 128],
                activation='norm_relu', use_skip_connections=False, dropout_rate=0.2, return_sequences=True,
                name='TCN_1')(m)
        #         m = Bidirectional(CuDNNGRU(rnnunit,return_sequences=True),name='blstm1')(m)
        #         m = Dense(rnnunit,name='blstm1_out',activation='linear')(m)
        #         m = Bidirectional(CuDNNGRU(rnnunit,return_sequences=True),name='blstm2')(m)

        #         m = Bidirectional(GRU(rnnunit, return_sequences=True), name='blstm1')(m)
        #         m = Dense(rnnunit, name='blstm1_out', activation='linear')(m)
        #         m = Bidirectional(GRU(rnnunit, return_sequences=True), name='blstm2')(m)

        y_pred = Dense(n_classes, name='blstm2_out', activation='softmax')(m)

        basemodel = Model(inputs=inputs, outputs=y_pred)

        return basemodel

    def _build_model(self):
        # 3. CTC Loss Model
        labels = Input(name='the_labels', shape=[None, ], dtype='float32')
        input_length = Input(name='input_length', shape=[1], dtype='int64')
        label_length = Input(name='label_length', shape=[1], dtype='int64')

        loss_out = Lambda(
            self._ctc_lambda_func,
            output_shape=(1,),
            name='ctc'
        )([self.basemodel.output, labels, input_length, label_length])

        model = Model(
            inputs=[self.basemodel.input, labels, input_length, label_length],
            outputs=[loss_out]
        )

        # Optimizer

        op = Adam(
            lr=0.001,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-08,
            decay=0.0,
            clipnorm=5
        )

        model.compile(
            loss={'ctc': lambda y_true, y_pred: y_pred},
            optimizer=op
        )

        test_func = keras.backend.function(
            [self.basemodel.input],
            [self.basemodel.output]
        )

        reshaped = keras.backend.reshape(input_length, (-1,))

        top_k_decoded, _ = ctc_decode(
            self.basemodel.output,
            reshaped,
            greedy=False,
            beam_width=5,
            top_paths=3
        )

        decode = keras.backend.function(
            [self.basemodel.output, reshaped],
            top_k_decoded
        )

        return model, test_func

    def __load_label_text(self, label_text_path):
        with open(label_text_path, "rb") as f:
            self.label_text = json.load(f)

    def load(self, weights_path, label_text_path):
        with self.graph.as_default():
            with self.session.as_default():
                self.model_json = label_text_path
                self.__load_label_text(self.model_json)
                self.model_weight = weights_path
                self.basemodel = self._build_basemodel(height=48, n_classes=4332)
                self.model, self.decode = self._build_model()
                self.model.load_weights(self.model_weight)

    def labels_to_text(self, labels):
        ret = []
        for c in labels:
            if c == len(self.label_text):  # CTC Blank
                ret.append("")
            else:
                ret.append(self.label_text[str(c)])
        return "".join(ret)

    def confident(self, out):
        out_tmp = out.reshape(-1, 4332)
        max_t, max_c = out_tmp.shape
        confidence_score = 0
        for t in range(max_t):
            max_idx = np.argmax(out_tmp[t, :])
            confidence_score = float(confidence_score) + float(out_tmp[t, max_idx])
        confidence_score = confidence_score / max_t
        return confidence_score

    def predict(self, img_file):
        with self.graph.as_default():
            with self.session.as_default():
                if isinstance(img_file, np.ndarray):
                    img = img_file
                    img = img.astype(float)
                    img = img / 255
                elif isinstance(img_file, str):
                    img = cv2.imread(img_file, 0)
                    img = img.astype(float)
                    img = img / 255
                if len(img.shape) > 2:
                    h, w, _ = img.shape
                else:
                    h, w = img.shape
                # resize input image into 48*width new image to predict
                if w == 0 or h == 0:
                    raise ValueError("Error in input image!")
                aspect_ratio = w / h
                resized_img = cv2.resize(img, (int(48 * aspect_ratio), 48))
                if len(resized_img.shape) > 2:
                    resized_height, resized_width, _ = resized_img.shape
                else:
                    resized_height, resized_width = resized_img.shape
                reshaped_img = resized_img.reshape(-1, resized_height, resized_width, 1)
                out = self.basemodel.predict(reshaped_img)
                ret = []
                confident_score = self.confident(out)
                for j in range(out.shape[0]):
                    out_best = list(np.argmax(out[j, 2:], 1))
                    out_best = [k for k, g in itertools.groupby(out_best)]
                    outstr = self.labels_to_text(out_best)
                    ret.append(outstr)
                data = {
                    'prediction': ret,
                    'confidence': confident_score
                }

                return ret

    def predict_batch(self, x_list, batch_size=8):
        with self.graph.as_default():
            with self.session.as_default():
                result_list = []
                for x in x_list:
                    out = self.basemodel.predict(x, batch_size=batch_size)
                    ret = []
                    confident_score = self.confident(out)
                    for j in range(out.shape[0]):
                        out_best = list(np.argmax(out[j, 2:], 1))
                        out_best = [k for k, g in itertools.groupby(out_best)]
                        outstr = self.labels_to_text(out_best)
                        ret.append(outstr)
                    result_list.append(ret)
                    data = {
                        'prediction': ret,
                        'confidence': confident_score
                    }

                return result_list

    def fit(self, x, y):
        pass

    def save(self, path):
        pass
