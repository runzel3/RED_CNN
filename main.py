import model
from utils import get_image_path, scale_image, imread, load_data, load_data_directly

IS_TRAINING = True
def main():
    if IS_TRAINING:
        my_model = model.RED_CNN()
        data, labels = load_data_directly()
        my_model.train(data, labels)
    else:
        my_model = model.RED_CNN()
        my_model.test()


if __name__ == '__main__':
    main()