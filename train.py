import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
MAX_FEATURES=10000  
MAX_LEN=500         
EMBEDDING_DIM=32    
RNN_UNITS=32        
EPOCHS=5
BATCH_SIZE=128
def train_model():
    print("Downloading and loading IMDB data...")
    (x_train,y_train),(x_test,y_test)=imdb.load_data(num_words=MAX_FEATURES)
    print(f"Padding sequences to {MAX_LEN} words...")
    x_train=pad_sequences(x_train,maxlen=MAX_LEN)
    x_test=pad_sequences(x_test,maxlen=MAX_LEN)
    print("Building Simple RNN model...")
    model=Sequential([
        Embedding(input_dim=MAX_FEATURES,output_dim=EMBEDDING_DIM),
        SimpleRNN(RNN_UNITS,return_sequences=False),
        Dense(1,activation='sigmoid') 
    ])
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    model.summary()
    print("Training model...")
    model.fit(
        x_train, 
        y_train, 
        epochs=EPOCHS, 
        batch_size=BATCH_SIZE, 
        validation_split=0.2
    )
    print("Evaluating model on test data...")
    loss,accuracy=model.evaluate(x_test, y_test)
    print(f"Test Accuracy: {accuracy:.4f}")
    print("Saving model to 'simple_rnn_imdb.h5'...")
    model.save('simple_rnn_imdb.h5')
    print("Training complete!")
if __name__ == "__main__":
    train_model()