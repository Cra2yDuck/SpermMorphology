import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import __init__

def plot_image(q, predictions_array, true_label, img):
    true_label, img = true_label[q], img[q]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(predicted_label,
                                         100*np.max(predictions_array),
                                         true_label),
               color=color)

def plot_value_array(q, predictions_array, true_label):
    true_label = true_label[q]
    plt.grid(False)
    plt.xticks(range(16))
    plt.yticks([])
    thisplot = plt.bar(range(16), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

def showing_all_shit(predictions_all):
    num_rows = 5
    num_cols = 3
    num_images = num_rows*num_cols
    plt.figure(figsize=(2*2*num_cols, 2*num_rows))
    for i in range(num_images):
        randi = np.random.randint(0, 300)
        plt.subplot(num_rows, 2*num_cols, 2*i+1)
        plot_image(randi, predictions_all[randi], lab_test, im_test_raw)
        plt.subplot(num_rows, 2*num_cols, 2*i+2)
        plot_value_array(randi, predictions_all[randi], lab_test)
    plt.tight_layout()
    plt.show()

def model_init(lay, lays):
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(128, 128)),
        tf.keras.layers.Dense(lays, activation='relu'),
        tf.keras.layers.Dense(lays/2, activation='relu'),
        tf.keras.layers.Dense(16),
    ])
    return model

def training(img_array, train_array, eps):
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.fit(im_train, lab_train, epochs=eps)

def input_normalization(im, lab):
    diff = 300
    counter = 0
    i = 0
    while counter < diff:
        if lab[i] == 0:
            lab = np.delete(lab, i, 0)
            im = np.delete(im, i, 0)
            counter += 1
            i-=1
        i += 1
    k = [0]*16
    for i in lab:
        k[i] += 1
    diff = k[0]-max(k[1:])
    print(f'Normalization completed, current diff is {diff}')
    return im, lab

path = 'mhsma-dataset-master/mhsma/'

im_train_raw, im_valid_raw, im_test_raw, \
lab_train, lab_valid, lab_test = __init__.init(path)

#im_train_raw = np.concatenate((im_train_raw, im_valid_raw, im_test_raw), axis=0)
#lab_train = np.concatenate((lab_train, lab_valid, lab_test), axis=0)

#im_train_raw, lab_train = input_normalization(im_train_raw, lab_train)

im_train = im_train_raw/255.0
im_valid = im_valid_raw/255.0
im_test = im_test_raw/255.0

lay, lays = 3, 512
eps = 9

model = model_init(lay, lays)
training(im_train, lab_train, eps)

valid_loss, valid_acc = model.evaluate(im_valid,  lab_valid, verbose=2)
print('\nTest accuracy:', valid_acc)

f = open('stat.txt', 'a')
output = f'Layers - {lay}; LaySize - {lays}; Epochs - {eps}; Acc - {valid_acc}\n'
print(output)
f.write(output)
f.close()

probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])

predictions = probability_model.predict(im_test)

showing_all_shit(predictions)