# text_summarisation_CNN-DailyMail


This project implements a text summarization model using deep learning in Python. The model is trained on a dataset of news articles and their corresponding summaries, and is able to generate summaries for new articles.

## Data set
we are using CNN daily mail data set, in particular i am using the test.csv(49 MB) that contains around 11k instances of data ,i am splitting this data to train and also to test,
source="https://www.kaggle.com/datasets/gowrishankarp/newspaper-text-summarization-cnn-dailymail" 

## Model Description

The model used for this project is a sequence-to-sequence model with attention layer(Transformer with encoder-decoder architecture)
### Visualisation of the data model
<img src="https://github.com/kartik5106/text_summarisation_CNN-DailyMail/blob/main/model.png" width="800" height= "900">  

components of the data model
* Encoder- Encoder_inputs is used in order to encode the words into numeric data for processing by LSTM layers.

 * LSTM Layers- We use 3 LSTM layers in order to process the data effectively. You can also experiment by adding or removing the layers in order to find more better accuracies. return_sequences in LSTM layer is set true until we want to add more layers consecutively.

  * Decoder- Decoder again converts the numeric data into the understandable word formats.

   *  Attention layer has been added to selectively focus on relevant information while discarding non-useful information. This is done by cognitively mapping the generated sentences with the inputs of the encoder layer.

  *   Dense layer has been added to mathematically represent the matrix-vector multiplication in neurons. It is used to change the dimensions of the vectors for processing between various layers.
  
###  Working of the model

This is a sequence-to-sequence model with attention mechanism, designed for text summarisation. The model consists of an encoder and a decoder. The encoder takes the input sequence and produces a hidden state representation. The decoder takes the hidden state representation and generates the output sequence.
The encoder uses an embedding layer followed by three LSTM layers with dropout and recurrent dropout for regularization. The first two LSTM layers return the sequence of hidden states, while the last LSTM layer returns the final hidden state and the cell state.
The decoder also uses an embedding layer and an LSTM layer with dropout and recurrent dropout. It takes the final hidden state and cell state from the encoder's last LSTM layer as initial states. Then, an attention layer is added to compute the attention weights between the encoder's hidden states and the decoder's hidden state. The attention weights are used to weight the encoder's hidden states, and the resulting context vector is concatenated with the decoder's hidden state to produce the final input to the dense layer.
Finally, a time-distributed dense layer with softmax activation is used to generate the output sequence. The model takes two inputs - the input sequence and the target sequence - and produces the output sequence as its output

## Results

{'rouge-1': {'r': 0.0, 'p': 0.0, 'f': 0.0},
 'rouge-2': {'r': 0.0, 'p': 0.0, 'f': 0.0},
 'rouge-l': {'r': 0.0, 'p': 0.0, 'f': 0.0}}
 
 #### Remarks:
This model is underperforming and i suspect it may be due to the sheer size of the data set i am using as it is well known for summarisers to fail when the data size is very large or some issue with the embeddings i have tried numerous ways to come up with ways to to improve the mode but i have not recieved any significant improvement.


# Model no 02
I have tried to implement another simpler text summeriser using only 'nltk' libraries(no machine learning models used)
In the context of text summarization, the algorithm is used to rank sentences in a document based on their importance.

following steps are carried out to achieve this:

* Fetching Articles from Wikipedia by data scraping
* Preprocessing 
* Tokenisation
* Find Weighted Frequency of Occurrence

Next we need to find the weighted frequency of occurrences of all the words. We can find the weighted frequency of each word by dividing its frequency by the frequency of the most occurring word. The following table contains the weighted frequencies for each word


To find the weighted frequency, we can simply divide the number of occurances of all the words by the frequency of the most occurring word

* Calculating Sentence Scores

We have now calculated the weighted frequencies for all the words. Now is the time to calculate the scores for each sentence by adding weighted frequencies of the words that occur in that particular sentence. We do not want very long sentences in the summary, therefore, we calculate the score for only sentences with less than 20 words 

* Getting the Summary

Now we have the sentence_scores dictionary that contains sentences with their corresponding score. To summarize the article, we can take top N sentences with the highest scores. The following script retrieves top 7 sentences and prints them on the screen.

we use the heapq library and call its nlargest function to retrieve the top 7 sentences with the highest scores.

