import numpy as np
import re 
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from sklearn.datasets import load_files
from nltk.stem import PorterStemmer
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt



def remove_stopwords(text):
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'[0-9]*', '', text)
    text = re.sub(r'[^a-zA-Z\s]+', '', text)
    stop_words = set(stopwords.words('english'))
    words = nltk.word_tokenize(text)
    filtered_words = [word for word in words if word.lower() not in stop_words]
    filtered_text = " ".join(filtered_words)
    
    return filtered_text

def porter_stemming(text):
    text=remove_stopwords(text)
    stemmer = PorterStemmer()
    # Removing punctuations and converting to lowercase
    text = text.translate(str.maketrans("", "", string.punctuation)).lower()
    words = text.split()
    # Applying Porter's Stemming on each word
    stemmed_words = [stemmer.stem(word) for word in words]
    stemmed_words= " ".join(stemmed_words)
    return stemmed_words

def snowball_stemming(text):
    text=remove_stopwords(text)
    stemmer = SnowballStemmer("english")
    # Removing punctuations and converting to lowercase
    text = text.translate(str.maketrans("", "", string.punctuation)).lower()
    words = text.split()
    # Applying Snowball Stemming on each word
    stemmed_words = [stemmer.stem(word) for word in words]
    stemmed_words= " ".join(stemmed_words)
    return stemmed_words

def wordnet_pos(word):
    """Map POS tag to first character used by WordNetLemmatizer"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {
        'CC': wn.NOUN,  # coordinating conjunction
        'CD': wn.NOUN,  # cardinal digit
        'DT': wn.NOUN,  # determiner
        'EX': wn.NOUN,  # existential there
        'FW': wn.NOUN,  # foreign word
        'IN': wn.NOUN,  # preposition/subordinating conjunction
        'JJ': wn.ADJ,   # adjective
        'JJR': wn.ADJ,  # adjective, comparative
        'JJS': wn.ADJ,  # adjective, superlative
        'LS': wn.NOUN,  # list marker
        'MD': wn.NOUN,  # modal
        'NN': wn.NOUN,  # noun, singular
        'NNS': wn.NOUN, # noun plural
        'NNP': wn.NOUN, # proper noun, singular
        'NNPS': wn.NOUN, # proper noun, plural
        'PDT': wn.NOUN, # predeterminer
        'POS': wn.NOUN, # possessive ending
        'PRP': wn.NOUN, # personal pronoun
        'PRP$': wn.NOUN,# possessive pronoun
        'RB': wn.ADV,   # adverb
        'RBR': wn.ADV,  # adverb, comparative
        'RBS': wn.ADV,  # adverb, superlative
        'RP': wn.NOUN,  # particle
        'TO': wn.NOUN,  # to
        'UH': wn.NOUN,  # interjection
        'VB': wn.VERB,  # verb base form
        'VBD': wn.VERB, # verb past tense
        'VBG': wn.VERB, # verb gerund/present participle
        'VBN': wn.VERB, # verb past participle
        'VBP': wn.VERB, # verb non-3rd person singular present
        'VBZ': wn.VERB, # verb 3rd person singular present
        'WDT': wn.NOUN, # wh-determiner
        'WP': wn.NOUN,  # wh-pronoun
        'WP$': wn.NOUN, # possessive wh-pronoun
        'WRB': wn.ADV   # wh-adverb
    }

    return tag_dict.get(tag, wn.NOUN)

def lemmatization(text):
    text=remove_stopwords(text)
    lemmatizer = WordNetLemmatizer()
    # Removing punctuations and converting to lowercase
    text = text.translate(str.maketrans("", "", string.punctuation)).lower()
    words = nltk.word_tokenize(text)
    # Applying lemmatization on each word
    lemmatized_words = [lemmatizer.lemmatize(word,wordnet_pos(word)) for word in words]
    lemmatized_words= " ".join(lemmatized_words)
    return lemmatized_words

def lemmatization_stemming(text):
    lemmatized_words=lemmatization(text)
    prerpoccedText=snowball_stemming(lemmatized_words)    
    return prerpoccedText



# Loading data
data = load_files('20_newsgroups', shuffle=True, encoding='utf-8', decode_error='replace')
X , y = data.data, data.target


# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing the data
for doc in range(len(X_train)):
    X_train[doc]=lemmatization_stemming(X_train[doc])
    
for doc in range(len(X_test)):
    X_test[doc]=lemmatization_stemming(X_test[doc])
    
vectorizer = TfidfVectorizer(min_df=0.0,max_df=0.9,strip_accents='ascii')
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

 

#**********************************************************************************************
# Train and evaluate a Support Vector Machine classifier
clf_svm = LinearSVC(C=1.0, max_iter=100000)
clf_svm.fit(X_train, y_train)
y_pred_svm = clf_svm.predict(X_test)
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print("Support Vector Machine Accuracy Score:", accuracy_svm)


#confusion_matrix
y_pred_svm = clf_svm.predict(X_test)
confusion_mat_svm = confusion_matrix(y_test, y_pred_svm)


#Visualize the confusion matrix
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(confusion_mat_svm)
ax.grid(False)
ax.set_xlabel('Predicted outputs', fontsize=12, color='black')
ax.set_ylabel('True outputs', fontsize=12, color='black')
ax.xaxis.set(ticks=np.arange(len(data.target_names)))
ax.yaxis.set(ticks=np.arange(len(data.target_names)))
ax.set_xticklabels(data.target_names, rotation=90)
ax.set_yticklabels(data.target_names)
for i in range(len(data.target_names)):
    for j in range(len(data.target_names)):
        ax.text(j, i, confusion_mat_svm[i, j], ha="center", va="center", color="white")

plt.show()
#**********************************************************************************************
# Train and evaluate a Multinomial Naive Bayes classifier
clf_nb = MultinomialNB(alpha=0.1)  
clf_nb.fit(X_train, y_train)
y_pred_nb = clf_nb.predict(X_test)
accuracy_nb = accuracy_score(y_test, y_pred_nb)
print("Multinomial Naive Bayes Accuracy Score:", accuracy_nb)


#confusion_matrix
y_pred_nb = clf_nb.predict(X_test)
confusion_mat_nb = confusion_matrix(y_test, y_pred_nb)



#Visualize the confusion matrix
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(confusion_mat_nb)
ax.grid(False)
ax.set_xlabel('Predicted outputs', fontsize=12, color='black')
ax.set_ylabel('True outputs', fontsize=12, color='black')
ax.xaxis.set(ticks=np.arange(len(data.target_names)))
ax.yaxis.set(ticks=np.arange(len(data.target_names)))
ax.set_xticklabels(data.target_names, rotation=90)
ax.set_yticklabels(data.target_names)
for i in range(len(data.target_names)):
    for j in range(len(data.target_names)):
        ax.text(j, i, confusion_mat_nb[i, j], ha="center", va="center", color="white")

plt.show()
#**********************************************************************************************
# # Train and evaluate a Random Forest classifier
clf_rf = RandomForestClassifier(n_estimators=100)
clf_rf.fit(X_train, y_train)
y_pred_rf = clf_rf.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print("Random Forest Accuracy Score:", accuracy_rf)


#confusion_matrix
y_pred_rf = clf_rf.predict(X_test)
confusion_mat_rf = confusion_matrix(y_test, y_pred_rf)


#Visualize the confusion matrix
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(confusion_mat_rf)
ax.grid(False)
ax.set_xlabel('Predicted outputs', fontsize=12, color='black')
ax.set_ylabel('True outputs', fontsize=12, color='black')
ax.xaxis.set(ticks=np.arange(len(data.target_names)))
ax.yaxis.set(ticks=np.arange(len(data.target_names)))
ax.set_xticklabels(data.target_names, rotation=90)
ax.set_yticklabels(data.target_names)
for i in range(len(data.target_names)):
    for j in range(len(data.target_names)):
        ax.text(j, i, confusion_mat_rf[i, j], ha="center", va="center", color="white")

plt.show()
#**********************************************************************************************
# Train a Decision Tree classifier
clf_dt = DecisionTreeClassifier()  
clf_dt.fit(X_train, y_train)
y_pred_dt = clf_dt.predict(X_test)
accuracy_dt = accuracy_score(y_test, y_pred_dt)
print("Decision Tree Accuracy Score:", accuracy_dt)


#confusion_matrix
y_pred_dt = clf_dt.predict(X_test)
confusion_mat_dt = confusion_matrix(y_test, y_pred_dt)



#Visualize the confusion matrix
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(confusion_mat_dt)
ax.grid(False)
ax.set_xlabel('Predicted outputs', fontsize=12, color='black')
ax.set_ylabel('True outputs', fontsize=12, color='black')
ax.xaxis.set(ticks=np.arange(len(data.target_names)))
ax.yaxis.set(ticks=np.arange(len(data.target_names)))
ax.set_xticklabels(data.target_names, rotation=90)
ax.set_yticklabels(data.target_names)
for i in range(len(data.target_names)):
    for j in range(len(data.target_names)):
        ax.text(j, i, confusion_mat_dt[i, j], ha="center", va="center", color="white")

plt.show()
#**********************************************************************************************
# Train a Logistic Regression classifier
clf_lr = LogisticRegression(max_iter=500)  
clf_lr.fit(X_train, y_train)
y_pred_lr = clf_lr.predict(X_test)
accuracy_lr = accuracy_score(y_test, y_pred_lr)
print("Logistic Regression Accuracy Score:", accuracy_lr)


#confusion_matrix
y_pred_lr = clf_lr.predict(X_test)
confusion_mat_lr = confusion_matrix(y_test, y_pred_lr)


#Visualize the confusion matrix
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(confusion_mat_lr)
ax.grid(False)
ax.set_xlabel('Predicted outputs', fontsize=12, color='black')
ax.set_ylabel('True outputs', fontsize=12, color='black')
ax.xaxis.set(ticks=np.arange(len(data.target_names)))
ax.yaxis.set(ticks=np.arange(len(data.target_names)))
ax.set_xticklabels(data.target_names, rotation=90)
ax.set_yticklabels(data.target_names)
for i in range(len(data.target_names)):
    for j in range(len(data.target_names)):
        ax.text(j, i, confusion_mat_lr[i, j], ha="center", va="center", color="white")

plt.show()
#**********************************************************************************************




input_text = """
 Electrical energy storage is required in many applications — telecommunication devices,
 such as cell phones and pagers, stand-by power systems, and electric/hybrid vehicles. 
 The specifications for the various energy storage devices are given in terms of energy 
 stored (W h) and maximum power (W) as well as size and weight, initial cost and life. 
 A storage device to be suitable for a particular application must meet all the requirements.
 As power requirements for many applications become more demanding, it is often reasonable to
 consider separating the energy and power requirements by providing for the peak power by 
 using a pulse power device (capacitor) that is charged periodically from a primary energy
 storage unit (battery). For applications in which significant energy is needed in pulse form,
 traditional capacitors as used in electronic circuits cannot store enough energy in the volume
 and weight available. For these applications, the development of high energy density capacitors
 (ultracapacitors or electrochemical capacitors) has been undertaken by various groups around the world.
 This paper considers in detail why such capacitors are being developed, how they function, 
 and the present status and projected development of ultracapacitor technology.
"""
preprocessed_text = vectorizer.transform([input_text])

# Predict the category of the input text
predicted_label = clf_svm.predict(preprocessed_text)[0]
predicted_category = data.target_names[predicted_label]

# Print the predicted category
print("Predicted category:", predicted_category)









