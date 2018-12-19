from keras import Input, layers, models, losses, optimizers, initializers, activations, callbacks
import os
import numpy as np
from PIL import Image
import utils
from utils import PATCH_SIZE

checkpoints_dir = os.path.join(os.getcwd(), "checkpoints")
checkpoint_best_path = os.path.join(checkpoints_dir, 'best_weights.hdf5')
checkpoint_last_path = os.path.join(checkpoints_dir, 'last_weights.hdf5')
old_checkpoints_dir = os.path.join(os.getcwd(), "checkpoints_num_filter_96")
old_checkpoint_last_path = os.path.join(checkpoints_dir, 'last_weights.hdf5')



outputs_dir = './outputs'
evaluate_dir = './evals'

class RED_CNN(object):
    def __init__(self, num_kernel_per_layer=96, num_kernel_last_layer=1, kernel_size=(5, 5), lr=0.0006):
        print("Initializing model...")
        self.total_num_epoch_to_train = 10000
        self.batch_size = 128

        self.num_kernel_per_layer = num_kernel_per_layer
        self.num_kernel_last_layer = num_kernel_last_layer
        self.kernel_size = kernel_size
        self.lr = lr

        gauss_initializer = initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)
        y_0 = Input(shape=(None, None, 1))  # adapt this if using `channels_first` image data format
        y_1 = layers.Conv2D(self.num_kernel_per_layer, self.kernel_size, activation='relu', padding='valid', kernel_initializer=gauss_initializer, name='y_1')(y_0)
        y_2 = layers.Conv2D(self.num_kernel_per_layer, self.kernel_size, activation='relu', padding='valid', kernel_initializer=gauss_initializer, name='y_2')(y_1)
        y_3 = layers.Conv2D(self.num_kernel_per_layer, self.kernel_size, activation='relu', padding='valid', kernel_initializer=gauss_initializer, name='y_3')(y_2)
        y_4 = layers.Conv2D(self.num_kernel_per_layer, self.kernel_size, activation='relu', padding='valid', kernel_initializer=gauss_initializer, name='y_4')(y_3)
        y_5 = layers.Conv2D(self.num_kernel_per_layer, self.kernel_size, activation='relu', padding='valid', kernel_initializer=gauss_initializer, name='y_5')(y_4)
        y_5_deconv = layers.Conv2DTranspose(self.num_kernel_per_layer, self.kernel_size, activation=None, padding='valid', kernel_initializer=gauss_initializer, name='y_5_deconv')(y_5)
        y4_add_y5deconv = layers.Add(name='y4_add_y5deconv')([y_4, y_5_deconv])
        y_6 = layers.Activation(activation='relu', name='y_6')(y4_add_y5deconv)
        y_7 = layers.Conv2DTranspose(self.num_kernel_per_layer, self.kernel_size, activation='relu', padding='valid', kernel_initializer=gauss_initializer, name='y_7')(y_6)
        y_7_deconv = layers.Conv2DTranspose(self.num_kernel_per_layer, self.kernel_size, activation=None, padding='valid', kernel_initializer=gauss_initializer, name='y_7_deconv')(y_7)
        y2_add_y7deconv = layers.Add(name='y2_add_y7deconv')([y_2, y_7_deconv])
        y_8 = layers.Activation(activation='relu', name='y_8')(y2_add_y7deconv)
        y_9 = layers.Conv2DTranspose(self.num_kernel_per_layer, self.kernel_size, activation='relu', padding='valid', kernel_initializer=gauss_initializer, name='y_9')(y_8)
        # Note: last layer only has 1 kernel
        y_9_deconv = layers.Conv2DTranspose(self.num_kernel_last_layer, self.kernel_size, activation=None, padding='valid', kernel_initializer=gauss_initializer, name='y_9_deconv')(y_9)
        y0_add_y9_deconv = layers.Add(name='y0_add_y9_deconv')([y_0, y_9_deconv])
        output = layers.Activation(activation='relu', name='stage1_output')(y0_add_y9_deconv)

        # stage2_y_1 = layers.Conv2D(self.num_kernel_per_layer, self.kernel_size, activation='relu', padding='valid',
        #                     kernel_initializer=gauss_initializer, name='stage2_y_1')(stage1_output)
        # stage2_y_2 = layers.Conv2D(self.num_kernel_per_layer, self.kernel_size, activation='relu', padding='valid',
        #                     kernel_initializer=gauss_initializer, name='stage2_y_2')(stage2_y_1)
        # stage2_y_3 = layers.Conv2D(self.num_kernel_per_layer, self.kernel_size, activation='relu', padding='valid',
        #                     kernel_initializer=gauss_initializer, name='stage2_y_3')(stage2_y_2)
        # stage2_y_4 = layers.Conv2D(self.num_kernel_per_layer, self.kernel_size, activation='relu', padding='valid',
        #                     kernel_initializer=gauss_initializer, name='stage2_y_4')(stage2_y_3)
        # stage2_y_5 = layers.Conv2D(self.num_kernel_per_layer, self.kernel_size, activation='relu', padding='valid',
        #                     kernel_initializer=gauss_initializer, name='stage2_y_5')(stage2_y_4)
        # stage2_y_5_deconv = layers.Conv2DTranspose(self.num_kernel_per_layer, self.kernel_size, activation=None,
        #                                     padding='valid', kernel_initializer=gauss_initializer, name='stage2_y_5_deconv')(stage2_y_5)
        # stage2_y4_add_y5deconv = layers.Add(name='stage2_y4_add_y5deconv')([stage2_y_4, stage2_y_5_deconv])
        # stage2_y_6 = layers.Activation(activation='relu', name='stage2_y_6')(stage2_y4_add_y5deconv)
        # stage2_y_7 = layers.Conv2DTranspose(self.num_kernel_per_layer, self.kernel_size, activation='relu', padding='valid',
        #                              kernel_initializer=gauss_initializer, name='stage2_y_7')(stage2_y_6)
        # stage2_y_7_deconv = layers.Conv2DTranspose(self.num_kernel_per_layer, self.kernel_size, activation=None,
        #                                     padding='valid', kernel_initializer=gauss_initializer, name='stage2_y_7_deconv')(stage2_y_7)
        # stage2_y2_add_y7deconv = layers.Add(name='stage2_y2_add_y7deconv')([stage2_y_2, stage2_y_7_deconv])
        # stage2_y_8 = layers.Activation(activation='relu', name='stage2_y_8')(stage2_y2_add_y7deconv)
        # stage2_y_9 = layers.Conv2DTranspose(self.num_kernel_per_layer, self.kernel_size, activation='relu', padding='valid',
        #                              kernel_initializer=gauss_initializer, name='stage2_y_9')(stage2_y_8)
        # # Note: last layer only has 1 kernel
        # stage2_y_9_deconv = layers.Conv2DTranspose(self.num_kernel_last_layer, self.kernel_size, activation=None,
        #                                     padding='valid', kernel_initializer=gauss_initializer, name='stage2_y_9_deconv')(stage2_y_9)
        # stage2_y0_add_y9_deconv = layers.Add(name='stage2_y0_add_y9_deconv')([stage1_output, stage2_y_9_deconv])
        #
        # stage2_output = layers.Activation(activation='relu', name='stage2_output')(stage2_y0_add_y9_deconv)

        self.model = models.Model(y_0, output)
        optimizer = optimizers.Adam(lr=lr)
        self.model.compile(loss=losses.mean_squared_error, optimizer=optimizer)
        self.current_epoch = 0
        if not os.path.exists(evaluate_dir):
            os.mkdir(evaluate_dir)

    def train(self, train_data, train_labels):
        print("Start training...")
        self.load_last_model()

        learning_rate_update_callback = callbacks.LearningRateScheduler(step_decay, verbose=1)
        checkpoint_last_callback = callbacks.ModelCheckpoint(checkpoint_last_path, verbose=1)
        checkpoint_best_only_callback = callbacks.ModelCheckpoint(checkpoint_best_path, verbose=1, save_best_only=True)

        my_prediction_callback = My_prediction_callback()
        callbacks_list = [checkpoint_last_callback, checkpoint_best_only_callback, my_prediction_callback, learning_rate_update_callback]
        self.model.fit(x=train_data, y=train_labels,
                       epochs=self.total_num_epoch_to_train,
                       batch_size=self.batch_size,
                       shuffle=True,
                       validation_split=0,
                       callbacks=callbacks_list)

    # def eval(self, noisy_img, save_name, clean_img=None):
    #     prediction = self.model.predict(np.array([noisy_img]))[0]
    #     prediction = prediction * 255
    #     prediction = prediction.astype('uint8').reshape((128, 128))
    #     predicted_img = Image.fromarray(prediction)
    #     save_name += "_" + str(self.current_epoch) + '.png'
    #     save_path = os.path.join(evaluate_dir, save_name)
    #     predicted_img.save(save_path)

    def test(self):
        self.load_last_model()
        saved_dir = os.path.join(os.getcwd(), "xray_images")
        saved_dir = os.path.join(saved_dir, "test_images_128x128")
        if not os.path.exists(saved_dir):
            os.mkdir(saved_dir)
        for i in range(1, 4000):
            if os.path.exists(utils.get_image_path(True, 64, i)):
                test_noisy_128 = utils.imread(utils.get_image_path(True, 64, i))
                test_noisy_128 = utils.scale_image(test_noisy_128, 2.0)  # Image size 128x128
                test_noisy_128 /= 255.0
                test_noisy_128 = test_noisy_128.reshape(128, 128, 1)

                prediction = self.model.predict(np.array([test_noisy_128]))[0]
                prediction = prediction * 255
                prediction = prediction.astype('uint8').reshape((128, 128))
                predicted_img = Image.fromarray(prediction)
                clean_image_path = utils.get_image_path(True, 128, i)
                predicted_img.save(clean_image_path)

    def load_best_model(self):
        print("[*] Reading checkpoint...")
        if not os.path.exists(checkpoints_dir):
            os.mkdir(checkpoints_dir)
            print("No checkpoint found.")
            return

        if not os.path.exists(checkpoint_best_path):
            print("No checkpoint found.")
            return
        self.model.load_weights(checkpoint_best_path)
        print("Load checkpoint successfully.")

    def load_last_model(self):
        print("[*] Reading checkpoint...")
        if not os.path.exists(checkpoints_dir):
            os.mkdir(checkpoints_dir)
            print("No checkpoint found.")
            return

        if not os.path.exists(checkpoint_last_path):
            print("No checkpoint found.")
            return
        self.model.load_weights(checkpoint_last_path)
        print("Load checkpoint successfully.")

    # def save_model(self):
    #     print("[*] Saving checkpoint... " + str(self.current_epoch))
    #     file_name = "checkpoint_" + format(self.current_epoch, "07") + ".h5"
    #     save_path = os.path.join(checkpoints_dir, file_name)
    #     self.model.save(save_path)


def step_decay(epoch):
   initial_lrate = 0.00005
   drop = 0.75
   epochs_drop = 5.0
   lrate = initial_lrate * np.power(drop,
           np.floor((1+epoch)/epochs_drop))
   lrate = max(lrate, 0.00001)
   return lrate

class My_prediction_callback(callbacks.Callback):
    def __init__(self):
        test_noisy_image = utils.imread(utils.get_image_path(False, 64, 4003))
        test_noisy_image = utils.scale_image(test_noisy_image, 2.0)  # Image size 128x128
        test_noisy_image /= 255.0
        test_noisy_image = test_noisy_image.reshape(128, 128, 1)
        self.noisy_img1 = test_noisy_image

        test_noisy_image = utils.imread(utils.get_image_path(False, 64, 19983))
        test_noisy_image = utils.scale_image(test_noisy_image, 2.0)  # Image size 128x128
        test_noisy_image /= 255.0
        test_noisy_image = test_noisy_image.reshape(128, 128, 1)
        self.noisy_img2 = test_noisy_image

    def on_epoch_end(self, epoch, logs={}):
        prediction = self.model.predict(np.array([self.noisy_img1]))[0]
        prediction = prediction * 255
        prediction = prediction.astype('uint8').reshape((128, 128))
        predicted_img = Image.fromarray(prediction)
        save_name = "img_4003_" + str(epoch) + '.png'
        save_path = os.path.join(evaluate_dir, save_name)
        predicted_img.save(save_path)

        prediction = self.model.predict(np.array([self.noisy_img2]))[0]
        prediction = prediction * 255
        prediction = prediction.astype('uint8').reshape((128, 128))
        predicted_img = Image.fromarray(prediction)
        save_name = "img_19983_" + str(epoch) + '.png'
        save_path = os.path.join(evaluate_dir, save_name)
        predicted_img.save(save_path)
