from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import tensorflow as tf


#'''
#Loading the training + test sets
(train_X, train_y), (test_X, test_y) = mnist.load_data()

#Normalizing the datasets x
train_Xn = tf.keras.utils.normalize(train_X, axis=1)
test_Xn = tf.keras.utils.normalize(test_X, axis=1)
#'''

#'''


model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28,28))) #28*28 noder, där varje nod är en pixel
model.add(tf.keras.layers.Dense(128,activation='relu'))
model.add(tf.keras.layers.Dense(128,activation='relu'))
model.add(tf.keras.layers.Dense(10,activation='softmax')) #output, 10 alternativ.

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
loss2 = 0
#for i in range(100):
i = 0
while True:
    i += 1
    print(i)
    model.fit(train_Xn,train_y, epochs=1) # is done on training data
    loss, accuracy = model.evaluate(test_Xn, test_y)  # Beräknar accuracyn för modellen

    if abs(loss - loss2) < 0.001:
        break
    else:
        loss2 = loss

#stochastic gradient descent
#backpropogation


model.save('Number.model')
#'''



#model = tf.keras.models.load_model('Number.model')

#'''
loss,accuracy = model.evaluate(test_Xn, test_y) #Beräknar accuracyn för modellen
print(loss)
print(accuracy)
#'''



#print(train_X[0]) #Print the array of an image
#plt.matshow(train_X[0]) #Print the plotted array of an imagep
#plt.show()




image_num = 1
while os.path.isfile(f"numbers/num{image_num}.png"):
    try:
        img = cv2.imread(f"numbers/num{image_num}.png")[:,:,0]
        img = np.invert(np.array([img]))
        prediction = model.predict(img)
        print("Det här numret är troligtvis: " + str(np.argmax(prediction)))
        plt.imshow(img[0],cmap=plt.cm.binary)
    except:
        print("Error")
    finally:
        image_num += 1