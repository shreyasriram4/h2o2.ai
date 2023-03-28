import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import auc
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from src.models.classifier import Classifier
from src.utils.file_util import FileUtil
from transformers import BertTokenizer
from transformers import BertConfig
from transformers import InputExample
from transformers import InputFeatures
from transformers import TFBertForSequenceClassification


class BERT(Classifier):
    def __init__(self, load_model=False):
        self.load_model = load_model
        self.saved_model_path = FileUtil.BERT_SENTIMENT_MODEL_DIR
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.bert_config = FileUtil.get_config()["BERT"]
        self.batch_size = self.bert_config["batch_size"]
        self.target_col = self.bert_config["target_col"]
        self.text_col = self.bert_config["text_col"]
        if not load_model:
            self.config_layer = BertConfig(
                hidden_dropout_prob=self.bert_config["dropout"])
            self.model = TFBertForSequenceClassification.from_pretrained(
                "bert-base-uncased",
                config=self.config_layer
                )
            self.optimizer = tf.keras.optimizers.Adam(
                learning_rate=self.bert_config["learning_rate"],
                epsilon=self.bert_config["epsilon"],
                clipnorm=self.bert_config["clipnorm"]
                )
            self.loss = tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True
                )
            self.metrics = tf.keras.metrics.SparseCategoricalAccuracy(
                'accuracy'
                )
            self.callback = tf.keras.callbacks.EarlyStopping(
                monitor=self.bert_config["callback_monitor"],
                patience=self.bert_config["callback_patience"],
                restore_best_weights=True
                )
        else:
            if not FileUtil.check_dir_exists(self.saved_model_path):
                raise FileNotFoundError("There is no saved model in path",
                                        self.saved_model_path)
            self.model = TFBertForSequenceClassification.from_pretrained(
                self.saved_model_path
                )

    def fit(self, train, valid):
        assert self.load_model is not True
        train_InputExamples, validation_InputExamples = \
            self.convert_data_to_examples(train, valid,
                                          self.text_col, self.target_col)

        train_data = self.convert_examples_to_tf_dataset(
            list(train_InputExamples),
            self.tokenizer
            )
        train_data = train_data.batch(self.batch_size)

        validation_data = self.convert_examples_to_tf_dataset(
            list(validation_InputExamples),
            self.tokenizer
            )
        validation_data = validation_data.batch(self.batch_size)

        self.model.compile(optimizer=self.optimizer,
                           loss=self.loss,
                           metrics=[self.metrics])

        history = self.model.fit(train_data,
                                 epochs=self.bert_config["epochs"],
                                 validation_data=validation_data,
                                 callbacks=[self.callback])

        return self.model, history

    def predict(self, test):
        tf_outputs = []
        for i in range(int(np.ceil(len(test) / self.batch_size))):
            tf_batch = self.tokenizer(
                list(test[self.text_col]
                     [i * self.batch_size: (i + 1) * self.batch_size]),
                max_length=50, padding=True, truncation=True,
                return_tensors='tf'
                )
            tf_outputs += list(self.model(tf_batch)["logits"])

        tf_predictions = tf.nn.softmax(tf_outputs, axis=-1)
        probs = list(map(lambda logit: max(np.exp(logit) / sum(np.exp(logit))),
                         tf_predictions))
        label = tf.argmax(tf_predictions, axis=1)
        label = label.numpy()
        return label, probs, tf_predictions

    def evaluate(self, valid):
        label, probs, tf_predictions = self.predict(valid)
        all_probs = list(map(lambda logit: np.exp(logit) / sum(np.exp(logit)),
                             tf_predictions))

        y_scores = list(map(lambda prob: prob[1], all_probs))
        precision, recall, thresholds = precision_recall_curve(
            valid[self.target_col], y_scores
            )

        ap = average_precision_score(valid[self.target_col], y_scores)
        pr_auc = auc(recall, precision)

        return ap, pr_auc

    def plot_training_acc_loss(self, history):
        losses = history.history['loss']
        accs = history.history['accuracy']
        val_losses = history.history['val_loss']
        val_accs = history.history['val_accuracy']
        epochs = len(losses)

        plt.figure(figsize=(12, 4))
        for i, metrics in enumerate(zip([losses, accs], [val_losses, val_accs],
                                        ['Loss', 'Accuracy'])):
            plt.subplot(1, 2, i + 1)
            plt.plot(range(1, epochs + 1), metrics[0],
                     label='Training {}'.format(metrics[2]))
            plt.title("BERT Training Graph")
            plt.plot(range(1, epochs + 1), metrics[1],
                     label='Validation {}'.format(metrics[2]))
            plt.title("BERT Training Graph")
            plt.legend()
        plt.show()
        plt.savefig(FileUtil.BERT_TRAINING_GRAPH_FILE_PATH)

    def convert_data_to_examples(self, train, test, DATA_COLUMN, LABEL_COLUMN):
        train_InputExamples = train.apply(
            lambda x: InputExample(guid=None,
                                   text_a=x[DATA_COLUMN],
                                   text_b=None,
                                   label=x[LABEL_COLUMN]),
            axis=1
            )

        validation_InputExamples = test.apply(
            lambda x: InputExample(guid=None,
                                   text_a=x[DATA_COLUMN],
                                   text_b=None,
                                   label=x[LABEL_COLUMN]),
            axis=1
            )

        return train_InputExamples, validation_InputExamples

    def convert_examples_to_tf_dataset(self, examples, tokenizer,
                                       max_length=50):
        features = []

        for e in examples:
            # Documentation is really strong for this method,
            # so please take a look at it
            input_dict = tokenizer.encode_plus(
                e.text_a,
                add_special_tokens=True,
                max_length=max_length,  # truncates if len(s) > max_length
                return_token_type_ids=True,
                return_attention_mask=True,
                pad_to_max_length=True,  # pads to the right by default
                truncation=True
            )

            input_ids, token_type_ids, attention_mask = (
                input_dict["input_ids"],
                input_dict["token_type_ids"],
                input_dict['attention_mask']
                )

            features.append(
                InputFeatures(
                    input_ids=input_ids, attention_mask=attention_mask,
                    token_type_ids=token_type_ids, label=e.label
                )
            )

        def gen():
            for f in features:
                yield (
                    {
                        "input_ids": f.input_ids,
                        "attention_mask": f.attention_mask,
                        "token_type_ids": f.token_type_ids,
                    },
                    f.label,
                )

        return tf.data.Dataset.from_generator(
            gen,
            ({"input_ids": tf.int32, "attention_mask": tf.int32,
              "token_type_ids": tf.int32}, tf.int64),
            (
                {
                    "input_ids": tf.TensorShape([None]),
                    "attention_mask": tf.TensorShape([None]),
                    "token_type_ids": tf.TensorShape([None]),
                },
                tf.TensorShape([]),
            ),
        )
