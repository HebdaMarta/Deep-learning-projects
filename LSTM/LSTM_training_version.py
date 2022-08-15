import numpy as np
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.datasets import fetch_20newsgroups
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import string
from nltk.corpus import stopwords
import re
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Embedding, LSTM, Bidirectional, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import itertools
from nltk.stem import WordNetLemmatizer


def remove_words(txt_lenghts, train_texts, train_target):
    x = np.array(train_texts)
    txt_filter = txt_lenghts < np.quantile(txt_lenghts, 0.9)
    x = x[txt_filter]
    y = train_target[txt_filter]
    return x, y

def clean_header(text):
    text = re.sub(r'(From:\s+[^\n]+\n)', '', text)
    text = re.sub(r'(Subject:[^\n]+\n)', '', text)
    text = re.sub(r'(([\sA-Za-z0-9\-]+)?[A|a]rchive-name:[^\n]+\n)', '', text)
    text = re.sub(r'(Last-modified:[^\n]+\n)', '', text)
    text = re.sub(r'(Version:[^\n]+\n)', '', text)
    return text

def clean_text(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

def lem_word(x):
    return [lem.lemmatize(x)]

def combine_text(list_of_text):
    combined_text = ' '.join(list_of_text)
    return combined_text


def plot_confusion_matrix(cm, class_names):
    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black
    threshold = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return figure

# Load data
train = fetch_20newsgroups(subset='train', shuffle=True)

train_texts = train.data
train_target = train.target

# Display some information about data
print(train_target[:10])

print(len(train_texts))
print(len(train_target))

print(train_texts[0])
print(train_target[0])
print(train.target_names[train_target[0]])

print(train.target_names)

df = pd.DataFrame(train_texts, columns=['mess'])
df['target'] = train_target
df['length'] = df['mess'].apply(len)
print(df.head())

# length of the texts
sns.barplot(x='target', y='length', data=df)
plt.show()

print(len(train_texts[0].split()))
txt_lenghts = np.array([len(txt.split(" ")) for txt in train_texts])
plt.hist(txt_lenghts, bins=50)
plt.show()

print(df['target'].value_counts())

# Mediana
plt.boxplot(txt_lenghts)
plt.show()
print(np.quantile(txt_lenghts, 0.9))

# Preprocessing

# Remove texts longer than 549
train_texts, train_target = remove_words(txt_lenghts, train_texts, train_target)
print(train_texts.shape, train_target.shape)

df = pd.DataFrame(train_texts, columns=['message'])
df['target'] = train_target

# Clean text

df['text_cleaned'] = df['message'].apply(clean_header)
df['cleaned_text'] = df['text_cleaned'].apply(lambda x: clean_text(x))

# Remove stopwords
stop_words = stopwords.words('english')
df['stopwords'] = df['cleaned_text'].str.split() \
    .apply(lambda x: ' '.join([word for word in x if word not in stop_words]))

# Lemmatization
lem = WordNetLemmatizer()
df['lemmatized_text'] = df['stopwords'].apply(lem_word)

df['final_text'] = df['lemmatized_text'].apply(lambda x: combine_text(x))

# Tokenizer
# max number of words in vocabulary
max_features = 6433

# max sentence length
max_len = 550

tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(df['final_text'].values)

# converting text into vectors
X = tokenizer.texts_to_sequences(df['final_text'].values)
X = pad_sequences(X, padding='post', maxlen=max_len)
Y = pd.get_dummies(df['target']).values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y)
print(X_train.shape, Y_train.shape)

print(df.head())

output_dim = 50

model = keras.Sequential()
model.add(Embedding(max_features, output_dim, input_length=X_train.shape[1], trainable=True))
model.add(Bidirectional(LSTM(units=256, return_sequences=True)))
model.add(Bidirectional(LSTM(units=256)))

model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(20, activation='softmax'))
model.summary()

batch_size = 32
model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
history = model.fit(X_train, Y_train, epochs=30, batch_size=batch_size, verbose=1, validation_data=(X_test, Y_test))

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

res = model.evaluate(X_test, Y_test, verbose=0)
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(res[0], res[1]))

# raport klasyfikacji
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)
_y_test = np.argmax(Y_test, axis=1)
print(classification_report(_y_test, y_pred, target_names=train.target_names))

cm = confusion_matrix(_y_test, y_pred)
plot_confusion_matrix(cm, train.target_names)
plt.show()

new_text = [
    "Senate Democrats are planning to hold the floor on Tuesday evening for an hours-long talk-a-thon on the issue of gun violence.The floor marathon comes as the White House is struggling to find a place to land in the weeks-long debate over potential gun-law reforms.“Many of my colleagues have seen their communities torn apart by gun violence; some by horrific mass shootings, others by a relentless, daily stream. Many of them have worked for years to bring commonsense gun safety measures before the Senate,” Senate Minority Leader Charles Schumer (D-N.Y.) said Tuesday, in announcing the plan from the Senate floor."]
seq = tokenizer.texts_to_sequences(new_text)
padded = pad_sequences(seq, maxlen=549)
pred = model.predict(padded)
labels = train.target_names
print(pred, labels[np.argmax(pred)])