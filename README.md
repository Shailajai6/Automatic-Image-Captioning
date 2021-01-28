# Image-Captioning
Designed and implemented a Neural Network architecture that describes the content of images and retrieves them solely based on the generated descriptions. The main objective of is caption generation for the image input. Used CNN and transfer learning for feature extraction from images and LSTM with attention for sequence of words as caption for the image. One of the important constraint of our project was to produce the image descriptions similar to human generated language. We were able to achieve that increase in information exchange in the generated descriptions.
Vocabulary Generation:
Generated the words from the image descriptions and stored in a dictionary. This vocabulary is used for training the embedding layers.
Word vectorization(Word embedding):
Used GloVe for word vectorization  for words in vocabulary generated. GloVe is an unsupervised learning algorithm for obtaining vector representations for words. Training is performed on aggregated global word-word co-occurrence statistics from a corpus, and the resulting representations showcase interesting linear substructures of the word vector space. 
Data Partition:
There are 8091 images in total in Flickr image dataset. Partitioned the data in to train and test. Training dataset has 6000 images and test dataset has 2091 images.
Flickr_8k.trainImages.txt :The training images used for training our model
Flickr_8k.testImages.txt :The test  images used for evaluating our model.
Encoded_test_images.pkl: The descriptions file used for the train images description was a pickle file.
Encoded_test_images.pkl: The descriptions file used for the test images description was a pickle file.

Model Description
Classic Image Captioning Model:
Encoder: For encoder we pass an image as input and preprocess it as per the model requirement. In our classic model we used CNN transfer learning pretrained model  Resnet50 for extracting the features of an image.  The output of the encoder would be the encoding vector of the image which will be reshaped as per the requirements of decoder.
Transfer Learning: Transfer learning is a technique where a model trained on a ImageNet dataset is re-purposed on the customized task. Transfer Learning is an optimization that allows rapid progress or improved performance when modelling the customized task.
Decoder: The input to the decoder is the output of the encoder and the other input is the word embeddings  of the vocabulary generated from image captions using LSTM. 
LSTM: Long short-term memory (LSTM) is an artificial recurrent neural network (RNN) architecture[1] used in the field of deep learning. Unlike standard feedforward neural networks, LSTM has feedback connections. It can not only process single data points (such as images), but also entire sequences of data (such as speech or video). It can process data sequentially and keep its hidden state through time.
 Then the two  inputs one is from LSTM cell state and the other is the feature vector image is concatenated . The concatenated output is passed through dense layer with relu as activation function and then it is passed to final dense layer which is designed to generate the captions from vocabulary with number of neurons equal to vocab_size.
Activation function used is SoftMax as  we are dealing with the vocabulary of words which are one-hot encoded.
Image Captioning using Attention mechanism:
We trained multi-model using attention mechanism with the combination of CNN and RNN Gating Recurrent unit.
Encoder:  The input to the encoder is image. We have used VGG16 a transfer learning technique for extracting the features from images . CNN acts as a feature extractor that compresses the information in the original image into a smaller representation. Since it encodes the content of the image into a smaller feature vector hence, this CNN is often called the encoder.
When we process this feature vector and use it as an initial input to the following RNN, then it would be called decoder because RNN would decode the process feature vector and turn it into natural language. We have used Teacher forcing
Technique where the target word is passed as the next input to the decoder.
With an Attention mechanism, the image is first divided into n parts, and we compute an image representation of each part. When the RNN is generating a new word, the attention mechanism is focusing on the relevant part of the image, so the decoder only uses specific parts of the image.

Bahadanau Attention:
In Bahdanau or Local attention, attention is placed only on a few source positions.  Local attention chooses to focus only on a small subset of the hidden states of the encoder per target word.
Local attention first finds an alignment position and then calculates the attention weight in the left and right windows where its position is located and finally weights the context vector. The main advantage of local attention is to reduce the cost of the attention mechanism calculation.
Design of Bahdanau Attention: 
All hidden states of the encoder and the decoder are used to generate the context vector. The attention mechanism aligns the input and output sequences, with an alignment score parameterized by a feed-forward network. It helps to pay attention to the most relevant information in the source sequence. The model predicts a target word based on the context vectors associated with the source position and the previously generated target words.

BLEU: 
To evaluate our captions in reference to the original caption we make use of an evaluation method called BLEU. It is used to analyze the correlation of n-gram between the translation statement to be evaluated and the reference translation statement. 
The advantage of BLEU is that the granularity it considers is an n-gram rather than a word, considering longer matching information. 
Prediction:
 We have saved the model weights in a folder using tf.keras.savemodel. Used the saved weights for predicting the caption for image input. In the prediction we used the saved CNN transfer learining model for extracting the features and then passed the feature vector to decoder. Then output caption is generated from the combination of the feature vector and the LSTM sequence of words.
Web Deployment : 
we used streamlit to deploy our model. Streamlit is an open-source app framework in python for machine learning and data science teams. 
