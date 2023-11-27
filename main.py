from pathlib import Path
import numpy as np
from typing import List
from tqdm.notebook import tqdm
from time import sleep
from PIL import Image
import IPython.display
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score
# import gdown

import torch

import tensorflow as tf
from tensorflow import keras
from keras import regularizers
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout, BatchNormalization, Rescaling
from keras.optimizers import Adam, Adamax
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.efficientnet import EfficientNetB0

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

EVALUATE_ONLY = True
TEST_ON_LARGE_DATASET = True
TISSUE_CLASSES = ('ADI', 'BACK', 'DEB', 'LYM', 'MUC', 'MUS', 'NORM', 'STR', 'TUM')


# DATASETS_LINKS = {
#     'train': '1XtQzVQ5XbrfxpLHJuL0XBGJ5U7CS-cLi',
#     'train_small': '1qd45xXfDwdZjktLFwQb-et-mAaFeCzOR',
#     'train_tiny': '1I-2ZOuXLd4QwhZQQltp817Kn3J0Xgbui',
#     'test': '1RfPou3pFKpuHDJZ-D9XDFzgvwpUBFlDr',
#     'test_small': '1wbRsog0n7uGlHIPGLhyN-PMeT2kdQ2lI',
#     'test_tiny': '1viiB0s041CNsAK4itvX8PnYthJ-MDnQc'
# }

class Dataset:
    def __init__(self, name):
        self.name = name
        self.is_loaded = False
        # url = f"https://drive.google.com/uc?export=download&confirm=pbef&id={DATASETS_LINKS[name]}"
        output = f'{name}.npz'
        # gdown.download(url, output, quiet=False)
        print(f'Loading dataset {self.name} from npz.')
        np_obj = np.load(f'{name}.npz')
        self.images = np_obj['data']
        self.labels = np_obj['labels']
        self.n_files = self.images.shape[0]
        self.is_loaded = True
        print(f'Done. Dataset {name} consists of {self.n_files} images.')

    def image(self, i):
        # read i-th image in dataset and return it as numpy array
        if self.is_loaded:
            return self.images[i, :, :, :]

    def images_seq(self, n=None):
        # sequential access to images inside dataset (is needed for testing)
        for i in range(self.n_files if not n else n):
            yield self.image(i)

    def random_image_with_label(self):
        # get random image with label from dataset
        i = np.random.randint(self.n_files)
        return self.image(i), self.labels[i]

    def random_batch_with_labels(self, n):
        # create random batch of images with labels (is needed for training)
        indices = np.random.choice(self.n_files, n)
        imgs = []
        for i in indices:
            img = self.image(i)
            imgs.append(self.image(i))
        logits = np.array([self.labels[i] for i in indices])
        return np.stack(imgs), logits

    def image_with_label(self, i: int):
        # return i-th image with label from dataset
        return self.image(i), self.labels[i]

    def get_info(self, set_name: str):  # LBL5
        # Pie chart
        unique_elements, counts = np.unique(self.labels, return_counts=True)
        pie_chart = []
        for element, count in zip(unique_elements, counts):
            # print(f"{element}: {count}")
            pie_chart.append(count)

        plt.pie(pie_chart, labels=['ADI', 'BACK', 'DEB', 'LYM', 'MUC', 'MUS', 'NORM', 'STR', 'TUM'],
                autopct='%1.1f', shadow=True)
        plt.title(f'{set_name}')
        # plt.savefig('models/train_small_pie.png')
        plt.show()

    def display_random_image(self):  # LBL5
        img, lbl = self.random_image_with_label()
        print()
        print(f'Got numpy array of shape {img.shape}, and label with code {lbl}.')
        print(f'Label code corresponds to {TISSUE_CLASSES[lbl]} class.')

        pil_img = Image.fromarray(img)
        pil_img.show()


class Metrics:
    @staticmethod
    def accuracy(gt: List[int], pred: List[int]):
        assert len(gt) == len(pred), 'gt and prediction should be of equal length'
        return sum(int(i[0] == i[1]) for i in zip(gt, pred)) / len(gt)

    @staticmethod
    def accuracy_balanced(gt: List[int], pred: List[int]):
        return balanced_accuracy_score(gt, pred)

    @staticmethod
    def print_all(gt: List[int], pred: List[int], info: str):
        print(f'metrics for {info}:')
        print('\t accuracy {:.4f}:'.format(Metrics.accuracy(gt, pred)))
        print('\t balanced accuracy {:.4f}:'.format(Metrics.accuracy_balanced(gt, pred)))


class Model:
    def __init__(self):
        self.input_shape = (224, 224, 3)

        self.model = self.build_model()
        self.hist = []
        self.epochs = 1

    def build_model(self):
        base_model = EfficientNetB0(include_top=False, weights="imagenet", input_shape=self.input_shape, pooling='max')
        base_model.trainable = False

        model = Sequential()

        model.add(base_model)
        model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001))
        model.add(Dense(256,
                        kernel_regularizer=regularizers.l2(l=0.016),
                        activity_regularizer=regularizers.l1(0.006),
                        bias_regularizer=regularizers.l1(0.006), activation='relu'))
        model.add(Dropout(rate=0.35))
        model.add(Dense(9, activation='softmax'))

        model.compile(Adamax(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

        model.summary()

        return model

    def save(self, name: str):
        # todo
        output = f'models/{name}.npz'
        self.model.save_weights(output, overwrite=True, save_format=None, options=None)
        # pass
        # example demonstrating saving the model to PROJECT_DIR folder on gdrive with name 'name'
        # arr = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        # np.savez(f'/content/drive/MyDrive/{name}.npz', data=arr)
        # weights = self.model.get_weights()
        # np.savez(f'{name}.npz', *weights)

    def load(self, name: str):
        # todo
        # —————————————
        input = f'models/{name}.npz'
        self.model.load_weights(input, skip_mismatch=False, by_name=False, options=None).expect_partial()
        # weights = np.load(f'{name}.npz')
        # self.model.set_weights(weights)
        # —————————————
        # example demonstrating loading the model with name 'name' from gdrive using link
        # name_to_id_dict = {
        #     'best': '1S8bwrVgvtSzadEX2aLlyb3VTlD31UI4R'
        # }
        # output = f'{name}.npz'
        # gdown.download(f'https://drive.google.com/uc?id={name_to_id_dict[name]}', output, quiet=False)
        # np_obj = np.load(f'{name}.npz')
        # print(np_obj['data'])

    def train(self, dataset: Dataset, n_epochs: int = 20, batch_size: int = 16):
        # you can add some plots for better visualization,
        # you can add model autosaving during training,
        # etc.
        self.epochs = n_epochs
        history = []

        print(f'Training started')

        generator = ImageDataGenerator(
            horizontal_flip=True,
            vertical_flip=True,
            validation_split=0.2,
        )  # LBL4: Использование аугментации и других способов синтетического расширения набора данных

        targets = to_categorical(dataset.labels, num_classes=9)  # Преобразование меток в категориальные

        # data, targets = dataset.random_batch_with_labels(dataset.n_files)  # Загрузка данных обучения dataset.n_files

        train_gen = generator.flow(dataset.images, targets,
                                   batch_size=batch_size,
                                   subset='training',
                                   shuffle=True)

        val_gen = generator.flow(dataset.images, targets,
                                 batch_size=batch_size,
                                 subset='validation',
                                 shuffle=True
                                 )

        hist = self.model.fit(train_gen, epochs=n_epochs, validation_data=val_gen)
        #   LBL1: Валидация модели на части обучающей выборки
        #   LBL2: Вывод различных показателей в процессе обучения

        history.append(hist)

        self.hist = history

    def plot_hist(self, name):  # LBL3: Построение графиков, визуализирующих процесс обучения
        if len(self.hist) == 1:
            plt.figure(figsize=(5, 3))

            hist = self.hist[0]

            acc = hist.history['accuracy']
            val_acc = hist.history['val_accuracy']
            loss = hist.history['loss']
            val_loss = hist.history['val_loss']

            epochs = range(1, len(acc) + 1)

            fig, axs = plt.subplots(2, 1, figsize=(5, 3 * 2))

            axs[0].plot(epochs, loss, 'bo', label='Training loss')
            axs[0].plot(epochs, val_loss, 'b', label='Validation loss')
            axs[0].set_title('Training and validation loss')
            axs[0].legend()

            axs[1].plot(epochs, acc, 'ro', label='Training acc')
            axs[1].plot(epochs, val_acc, 'r', label='Validation acc')
            axs[1].set_title('Training and validation accuracy')
            axs[1].legend()

            plt.tight_layout()

            # plt.plot(hist.history['loss'], color='red', label='loss')
            # plt.plot(hist.history['val_loss'], color='blue', label='val_loss')
        else:
            fig, axs = plt.subplots(len(self.hist), 1, figsize=(5, 3 * len(self.hist)))

            for i in range(len(self.hist)):
                loss_i = self.hist[i].history['loss']
                val_loss_i = self.hist[i].history['val_loss']

                axs[i].plot(np.arange(1, self.epochs + 1), np.log(loss_i), label=f'loss_{i + 1}')
                axs[i].plot(np.arange(1, self.epochs + 1), np.log(val_loss_i), label=f'val_loss_{i + 1}')

                axs[i].set_title(f'Log Loss on train and val sets. CV #{i + 1}')
                axs[i].legend(loc='upper right')

            plt.tight_layout()

        plt.savefig(f'models/{name}.png')
        plt.close()
        # plt.show()

    def test_on_dataset(self, dataset: Dataset, limit=None):
        # you can upgrade this code if you want to speed up testing using batches
        # predictions = []
        n = dataset.n_files if not limit else int(dataset.n_files * limit)
        # for img in tqdm(dataset.images_seq(n), total=n):
        #     predictions.append(self.test_on_image(img))
        # for img in dataset.images_seq(n):
        #     predictions.append(self.test_on_image(img))
        predictions = self.model.predict(dataset.images_seq(n), steps=n)
        predictions = np.argmax(predictions, axis=1)
        return predictions

    def test_on_image(self, img: np.ndarray):
        # todo: replace this code
        # prediction = np.random.randint(9)
        # sleep(0.05)
        # return prediction
        # prediction = np.argmax(self.model.predict(np.expand_dims(img, axis=0)))
        prediction = self.model.predict(np.expand_dims(img, axis=0))
        prediction = np.argmax(prediction[0])
        # sleep(0.05)
        return prediction


set_name = 'small'

if set_name == 'tiny':
    d_train = Dataset('train_tiny')
    d_test = Dataset('test_tiny')
elif set_name == 'small':
    d_train = Dataset('train_small')
    d_test = Dataset('test_small')
elif set_name == 'normal':
    d_train = Dataset('train')
    d_test = Dataset('test')

# d_train.get_info('Train')
# d_test.get_info('Test')
#
#
# d_train.display_random_image()

# Model initialization
# model = Model()
# n_epochs = 10
# filename_save = f'{set_name}/{set_name}_{n_epochs}epochs/{set_name}_{n_epochs}epochs'
# if not EVALUATE_ONLY:
#     model.train(d_train, n_epochs=n_epochs, batch_size=8)
#     model.plot_hist(f'{filename_save}_AccLoss_plot')
#     model.save(filename_save)
# else:
#     # todo: your link goes here
#     model.load(filename_save)


# # evaluating model on 10% of test dataset
# pred_1 = model.test_on_dataset(d_test_small, limit=0.1)
# Metrics.print_all(d_test_small.labels[:len(pred_1)], pred_1, '10% of test')

# # evaluating model on full test dataset (may take time)
# if TEST_ON_LARGE_DATASET:
#     pred_2 = model.test_on_dataset(d_test)
#     Metrics.print_all(d_test.labels, pred_2, 'test')


# Check the accuracy of the model in all test set
final_model = Model()
final_model.load('best/best')
d_test = Dataset('test')
d_test.get_info('Test')
pred = final_model.test_on_dataset(d_test)
Metrics.print_all(d_test.labels, pred, 'test')
