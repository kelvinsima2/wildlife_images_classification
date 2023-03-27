# wildlife_images_classification
This project classifies 90 classes of wild animals using neural networks and the pre-trained model InceptionResNetV2. The link to the publicly available data can be found [here](https://www.kaggle.com/datasets/iamsouravbanerjee/animal-image-dataset-90-different-animals).

# Data Preparation
Data was prepared as detailed in the [code](https://github.com/kelvinsima2/wildlife_images_classification/blob/main/Wildlife_Classification.ipynb). The image data was stored in a folder on Google Drive. The images were already separated into 90 folders representing different classes. 

# Model
The deep learning framework used in this project is Tensorflow. The model is summarized as follows (the base model is the InceptionResNetV2 found [here](https://keras.io/api/applications/inceptionresnetv2/)): 
* inputs = tf.keras.Input(shape=IMG_SHAPE)
* x = data_augmentation(inputs)
* x = preprocess_input(x)
* x = base_model(x, training=False)
* x = tf.keras.layers.BatchNormalization(renorm=True)(x)
* x = tf.keras.layers.GlobalAveragePooling2D()(x)
* x = tf.keras.layers.Dropout(0.5)(x)
* outputs = tf.keras.layers.Dense(90, activation='softmax')(x)
* model = tf.keras.Model(inputs, outputs)

# Results
Overall, the testing accuracy for the model was 90.6%. The training and validation accuracy and loss graphs are shown below: <br />
![Accuracy and Loss Graphs](/images/accuracy_wildlife.png)

<br />

As another test, a random picture of a kangaroo was obtained from the animals.py library. This was tested on the model and it correctly predicted the class, as shown below: <br />
![kangaroo](/images/kangaroo.png)




