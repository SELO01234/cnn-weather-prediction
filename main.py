from preprocess import *
import numpy as np
from model import ImageModel
from tensorflow.keras import utils
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools
from tensorflow.keras import utils as image_utils
import tensorflow as tf
if __name__ == '__main__':

    """Preprocess"""
    # process_all_images_overridden()

    """
    Training and confusion matrix 
    """
    # train_data = np.load(os.path.join('./prepared_dataset/', "train_data.npy"))
    # print(train_data.shape)
    #
    # train_label = np.load(os.path.join('./prepared_dataset/', "train_label.npy"))
    # print(train_label.shape)
    # print(train_label[0:31])
    # # 0 sunny, 1 rainy, 2 snowy, 3 foggy?
    #
    # validation_data = train_data[10681:11681] #1000
    # test_data = train_data[11681:12182] #500
    # train_data = train_data[:10680]
    #
    # validation_label = train_label[10681:11681] #1000
    # test_label = train_label[11681:12182] #500
    # train_label = train_label[:10680]
    #
    # train_label = utils.to_categorical(train_label, num_classes=4)
    # validation_label = utils.to_categorical(validation_label, num_classes=4)
    # test_label = utils.to_categorical(test_label, num_classes=4)
    #
    # imageModel = ImageModel()
    # imageModel.build()
    # imageModel.summary()
    # imageModel.compile()
    # imageModel.fit(train_data, train_label, validation_data, validation_label)
    #
    # predictions = imageModel.predict(test_data)
    #
    # print(np.round(predictions))
    #
    #
    # def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    #     plt.imshow(cm, interpolation='nearest', cmap=cmap)
    #     plt.title(title)
    #     plt.colorbar()
    #     tick_marks = np.arange(len(classes))
    #     plt.xticks(tick_marks, classes, rotation=45)
    #     plt.yticks(tick_marks, classes)
    #
    #     if normalize:
    #         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    #         print('Normalized confusion matrix')
    #     else:
    #         print('Confusion matrix')
    #
    #     print(cm)
    #
    #     thresh = cm.max() / 2
    #     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    #         plt.text(j, i, cm[i, j], horizontalalignment='center', color='black')
    #
    #     plt.tight_layout()
    #     plt.ylabel('True label')
    #     plt.xlabel('Predicted label')
    #
    # test_label_classes = np.argmax(test_label, axis=1)
    # predictions_classes = np.argmax(predictions, axis=1)
    #
    # cm = confusion_matrix(y_true=test_label_classes, y_pred=predictions_classes)
    #
    # cm_plot_labels = ['sunny', 'rainy', 'snowy', 'foggy']
    # plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')
    #
    # plt.show()

    """
    Predicting new image
    """
    def preprocess_image(img_path):
        img = image_utils.load_img(img_path, target_size=(100, 100))  # Ensure the target size matches your model's input
        img_array = image_utils.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        return img_array

    imageModel = tf.keras.models.load_model('cnn_model.h5')

    img_path = 'foggy_deneme2.jpg'  # Replace with your image path
    img_array = preprocess_image(img_path)

    # Make the prediction
    predictions = imageModel.predict(img_array)

    print(np.round(predictions))
