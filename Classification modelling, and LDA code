###Instructions
##Note: This code is divided into 2 parts - Part 1 will go upt making the total list of all recommendations based on referencing to Amazon. Part 2 will focus on Topic modelling, identifying and sorting the product recommendations for each topic.
##install geopy in python packages if pycharm or pip install geopy
#You will need the following files downloaded in your environment before running the code,
# 1. 'Craigslist_Data.csv',
# 2. 'worldcities.csv'

#While you run the code, you would also able to save some output files  (you may change the output file address)
#you will get following output files after running the code -
# 1. 'Coeff_Comp.csv' - Gives distance coefficients measures across 4 methods for all listings of craigslist,
# 2. 'combined_df_2.xlsx' - Combined list of Craigslist and Amazon with predicted classification values for all Craigslist products,
# 3. 'recommendations_sorted.xlsx' - Sorted list of topic wise recommendation of all 10 topics,
# 4. "topic_wise_recommendations.xlsx" - Final list of 10 recommendations for each topic for TV search in Indianapolis.

##Part A
# Importing the required libraries
import csv
import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import jaccard_score
from scipy.spatial.distance import cosine

# Reading the csv files
# df_amazon = pd.read_csv(amazon_data)
df_cl = pd.read_csv('Craigslist_Data.csv') #Combined scraped list of Craigslist and Amazon

df_amazon = df_cl[df_cl['Label'] == 'Amazon']
df_craigslist = df_cl[df_cl['Label'] == 'Craigslist']

print(len(df_craigslist))

# Converting the required columns to lists
list_amazon_pn = df_amazon['Product Name'].tolist()    #pn = product name
list_craigslist_pn = df_craigslist['Product Name'].tolist()

list_amazon_pd = df_amazon['Description'].tolist()    #pd = product description
list_craigslist_pd = df_craigslist['Description'].tolist()

list_craigslist_actual = df_craigslist['Actual Class']      # TV = 1, non-TV = 0

#list_amazon_pc = df_amazon['Category'].tolist()     # pc = product category

# Checking the number of elements in each list
print("Amazon list has " + str(len(list_amazon_pn)) + " products.")
print("CL_Appliance has " + str(len(list_craigslist_pn)) + " products.")

# Merging the required files for preprocessing
# Pre-processing includes tokenization, removing stop words & punctuations, lemmatization
# This step can be done separately for Amazon and CL lists

# Tokenizing product name files:
token_amazon_pn = []
for element in list_amazon_pn:
    tokenizer = nltk.word_tokenize(element)
    token_amazon_pn.append(tokenizer)
print(token_amazon_pn)      # the output is a list of lists

token_cl_pn = []
for element in list_craigslist_pn:
    tokenizer_2 = nltk.word_tokenize(element)
    token_cl_pn.append(tokenizer_2)
print(token_cl_pn)      # the output is a list of lists

# Tokenizing product description files
token_amazon_pd = []
for element in list_amazon_pd:
    tokenizer_3 = nltk.word_tokenize(element)
    token_amazon_pd.append(tokenizer_3)
print(token_amazon_pd)      # the output is a list of lists

print(len(list_craigslist_pd))

token_cl_pd = []
for element in list_craigslist_pd:
    tokenizer_4 = nltk.word_tokenize(element)
    token_cl_pd.append(tokenizer_4)
print(token_cl_pd)   # the output is a list of lists

# Removing stopwords and punctuations
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')
print(stopwords)

# From Amazon Product Names and Descriptions
sw_amazon_pn = []       # sw = stopwords
for list in token_amazon_pn:
    new_list = []
    for element in list:
        if element.isalpha():
            if not element in stopwords.words('english'):
                new_list.append(element)
    sw_amazon_pn.append(new_list)
print(sw_amazon_pn[0])

sw_amazon_pd = []       # sw = stopwords
for list in token_amazon_pd:
    new_list = []
    for element in list:
        if element.isalpha():
            if not element in stopwords.words('english'):
                new_list.append(element)
    sw_amazon_pd.append(new_list)
print(sw_amazon_pd[0])

# From Craigslist Product Names and Descriptions

sw_cl_pn = []       # sw = stopwords
for list in token_cl_pn:
    new_list = []
    for element in list:
        if element.isalpha():
            if not element in stopwords.words('english'):
                new_list.append(element)
    sw_cl_pn.append(new_list)
print(sw_cl_pn[0])

sw_cl_pd = []       # sw = stopwords
for list in token_cl_pd:
    new_list = []
    for element in list:
        if element.isalpha():
            if not element in stopwords.words('english'):
                new_list.append(element)
    sw_cl_pd.append(new_list)
print(sw_cl_pd[0])

# Lemmatizing the words in each list
nltk.download('wordnet')        # dowloading the wordnets

# Amazon Product Names and Descriptions
lemmatizer = nltk.stem.WordNetLemmatizer()
lem_amazon_pn = []
for list in sw_amazon_pn:
    new_list_2 = []
    for element in list:
        new_list_2.append(lemmatizer.lemmatize(element))
    lem_amazon_pn.append(new_list_2)
print(lem_amazon_pn)

lemmatizer2 = nltk.stem.WordNetLemmatizer()
lem_amazon_pd = []
for list in sw_amazon_pd:
    new_list_2 = []
    for element in list:
        new_list_2.append(lemmatizer2.lemmatize(element))
    lem_amazon_pd.append(new_list_2)
print(lem_amazon_pd[0])

# Craigslist Product Names and Descriptions

lemmatizer3 = nltk.stem.WordNetLemmatizer()
lem_cl_pn = []
for list in sw_cl_pn:
    new_list_3 = []
    for element in list:
        new_list_3.append(lemmatizer3.lemmatize(element))
    lem_cl_pn.append(new_list_3)
print(lem_cl_pn)

lemmatizer4 = nltk.stem.WordNetLemmatizer()
lem_cl_pd = []
for list in sw_cl_pd:
    new_list_4 = []
    for element in list:
        new_list_4.append(lemmatizer4.lemmatize(element))
    lem_cl_pd.append(new_list_4)
print(lem_cl_pd)

# Combining the Amazon and CL lists to form a vocabulary

# Doing it for Product Names
amazon_cl_pn = []
for list in lem_amazon_pn:
    random_string = ''
    for element in list:
        if list.index(element) == 0:
            random_string += element
        else:
            random_string += " " + element
    amazon_cl_pn.append(random_string)
print(len(amazon_cl_pn))

for list in lem_cl_pn:
    random_string = ''
    for element in list:
        if list.index(element) == 0:
            random_string += element
        else:
            random_string += " " + element
    amazon_cl_pn.append(random_string)
print(len(amazon_cl_pn))

# Doing it for Product Descriptions
amazon_cl_pd = []
for list in lem_amazon_pd:
    random_string = ''
    for element in list:
        if list.index(element) == 0:
            random_string += element
        else:
            random_string += " " + element
    amazon_cl_pd.append(random_string)
print(len(amazon_cl_pd))

for list in lem_cl_pd:
    random_string = ''
    for element in list:
        if list.index(element) == 0:
            random_string += element
        else:
            random_string += " " + element
    amazon_cl_pd.append(random_string)
print(len(amazon_cl_pd))

# Creating a vocabulary and a list of words framework

# For Product Names
vectorizer = TfidfVectorizer()
tfidf_matrix_pn = vectorizer.fit_transform(amazon_cl_pn)

print("Array of Craigslist(849) and Amazon(10) product names combined: ", tfidf_matrix_pn.shape)

tfidf_pn_df = pd.DataFrame(tfidf_matrix_pn.toarray(),columns = vectorizer.get_feature_names_out())


# For Product Descriptions
vectorizer2 = TfidfVectorizer()
tfidf_matrix_pd = vectorizer2.fit_transform(amazon_cl_pd)

print("Array of Craigslist(849) and Amazon(10) product docs combined: ", tfidf_matrix_pd.shape)


# Separating the matrix again into Amazon and Craigslist matrices
# Why? - We combined both to ensure they have the same number of dimensions.
# Now that the dimensions are taken care of, we evaluate the Euclidean distance.

# Doing this for names first

pn_array = tfidf_matrix_pn.toarray()
print(type(pn_array[0]))

pn_amazon_array = pn_array[:10]     # The first ten elements are Amazon
pn_cl_array = pn_array[10:]         # All the others are from Craigslist
print(pn_amazon_array.shape)
print(pn_cl_array.shape)

# Replicating this for product Descriptions

pd_array = tfidf_matrix_pd.toarray()

pd_amazon_array = pd_array[:10]
print(pd_amazon_array.shape)

pd_cl_array = pd_array[10:]
print(pd_cl_array.shape)

# Computing the Euclidean distance between the Amazon arrays and the CL arrays
# For Product Names

distance_matrix_pn = np.zeros((pn_amazon_array.shape[0],pn_cl_array.shape[0]))

for i in range(pn_amazon_array.shape[0]):
    for j in range(pn_cl_array.shape[0]):
        distance_matrix_pn[i, j] = np.linalg.norm(pn_amazon_array[i] - pn_cl_array[j])

print(distance_matrix_pn.shape)

distance_matrix_pn_T = distance_matrix_pn.T     # Creating a Transpose for better visualization

print(distance_matrix_pn_T.shape)

# For Product Descriptions

distance_matrix_pd = np.zeros((pd_amazon_array.shape[0],pd_cl_array.shape[0]))

for i in range(pd_amazon_array.shape[0]):
    for j in range(pd_cl_array.shape[0]):
        distance_matrix_pd[i, j] = np.linalg.norm(pd_amazon_array[i] - pd_cl_array[j])

print(distance_matrix_pd.shape)

distance_matrix_pd_T = distance_matrix_pd.T

print(distance_matrix_pd_T.shape)

# Getting the minimum value for each row

least_distance_pn = np.min(distance_matrix_pn_T, axis=1)
print(len(least_distance_pn))

least_distance_pd = np.min(distance_matrix_pd_T, axis=1)
print(least_distance_pd)

mean_pd = np.mean(least_distance_pd)
median_pd = np.median(least_distance_pd)

print("The mean is " + str(mean_pd) + ".")
print("The median is " + str(median_pd) + ".")

least_distance_pd_list = least_distance_pd.tolist()

# Computing other coefficients on the Product Descriptions

threshold_amazon = np.percentile(pd_amazon_array, 75)  # 75th percentile for Amazon vectors
threshold_craigslist = np.percentile(pd_cl_array, 75)  # 75th percentile for Craigslist vectors

# Convert TF-IDF vectors to binary using the thresholds
binary_amazon_vectors = np.where(pd_amazon_array > threshold_amazon, 1, 0)
binary_craigslist_vectors = np.where(pd_cl_array > threshold_craigslist, 1, 0)

# Initialize matrices to store the coefficients
dice_coeffs = np.zeros((len(pd_cl_array), len(pd_amazon_array)))
jaccard_coeffs = np.zeros((len(pd_cl_array), len(pd_amazon_array)))
cosine_sims = np.zeros((len(pd_cl_array), len(pd_amazon_array)))
overlap_coeffs = np.zeros((len(pd_cl_array), len(pd_amazon_array)))

# Compute the coefficients for each pair of vectors
for i, craigslist_vector in enumerate(binary_craigslist_vectors):
    for j, amazon_vector in enumerate(binary_amazon_vectors):
        # Intersection and Union for binary vectors
        intersection = np.sum(craigslist_vector & amazon_vector)
        union = np.sum(craigslist_vector | amazon_vector)

        # Compute Dice's Coefficient
        dice_coeffs[i, j] = 2 * intersection / (np.sum(craigslist_vector) + np.sum(amazon_vector))

        # Compute Jaccard's Coefficient
        jaccard_coeffs[i, j] = intersection / union if union != 0 else 0

        # Compute Cosine Similarity (using original TF-IDF vectors)
        cosine_sims[i, j] = 1 - cosine(pd_cl_array[i], pd_amazon_array[j])

        # Compute Overlap Coefficient
        overlap_coeffs[i, j] = intersection / np.min([np.sum(craigslist_vector), np.sum(amazon_vector)]) if np.min([np.sum(craigslist_vector), np.sum(amazon_vector)]) != 0 else 0

print(dice_coeffs)
print(jaccard_coeffs)
print(cosine_sims)
print(overlap_coeffs)

min_dice_coeffs = np.min(dice_coeffs, axis=1)
min_jaccard_coeffs = np.min(jaccard_coeffs, axis=1)
min_cosine_sims = np.min(cosine_sims, axis=1)
min_overlap_coeffs = np.min(overlap_coeffs, axis=1)

min_dice_coeffs_list = min_dice_coeffs.tolist()
min_jaccard_coeffs_list = min_jaccard_coeffs.tolist()
min_cosine_sims_list = min_cosine_sims.tolist()
min_overlap_coeffs_list = min_overlap_coeffs.tolist()

# Compiling the Craigslist product names, descriptions, and least distances

all_coeffs_combined_list = [[list_craigslist_pn[i], list_craigslist_pd[i], least_distance_pd_list[i],min_dice_coeffs_list[i],min_jaccard_coeffs_list[i],min_cosine_sims_list[i],min_overlap_coeffs_list[i],list_craigslist_actual[i]] for i in range(len(list_craigslist_pn))]

# Writing to the CSV file for all coefficients for all Craigslist vectors

csv_path_all_coeffs = 'Coeff_Comp.csv'

with open(csv_path_all_coeffs, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)

    # Optionally write headers
    writer.writerow(['Product Name', 'Product Description', 'Least Vector Distance', 'Dice Coeff', 'Jaccard Coeff', 'Cosine Coeff', 'Overlap Coeff', 'Actual Class'])

    # Write the data
    writer.writerows(all_coeffs_combined_list)

print(f"Data written to {csv_path_all_coeffs}")

new_output_file_2 = 'Coeff_Comp.csv'

df6 = pd.read_csv(new_output_file_2, encoding='utf-8')


# Evaluating the mode for Least Vector Distance

X1 = df6[['Least Vector Distance']]  # least vector distance as X
y = df6['Actual Class']   # Category as Y

# Split the dataset into training and testing sets (80-20 split)
X1_train, X1_test, y_train, y_test = train_test_split(X1, y, test_size=0.2, random_state=7)

# Initialize the RandomForestClassifier
model1 = RandomForestClassifier()

# Perform 5-fold cross-validation
cv_scores = cross_val_score(model1, X1_train, y_train, cv=5)

# Print the cross-validation scores
print("Cross-validation scores:", cv_scores)
print("Mean CV Score:", np.mean(cv_scores))

# Train the model on the entire training data
model1.fit(X1_train, y_train)

# Evaluate the model on the test set
test_score_vector_distance = model1.score(X1_test, y_test)
print("Test Score for Vector Distance:", test_score_vector_distance)

# Dice Coeff

X2 = df6[['Dice Coeff']]  # Dice Coefficient as X
y = df6['Actual Class']   # Category as Y

# Split the dataset into training and testing sets (80-20 split)
X2_train, X2_test, y_train, y_test = train_test_split(X2, y, test_size=0.2, random_state=7)

# Initialize the RandomForestClassifier
model2 = RandomForestClassifier()

# Perform 5-fold cross-validation
cv_scores = cross_val_score(model2, X2_train, y_train, cv=5)

# Print the cross-validation scores
print("Cross-validation scores:", cv_scores)
print("Mean CV Score:", np.mean(cv_scores))

# Train the model on the entire training data
model2.fit(X2_train, y_train)

# Evaluate the model on the test set
test_score_dice = model2.score(X2_test, y_test)
print("Test Score for Dice Coefficient:", test_score_dice)

# Jaccard Coeff

X3 = df6[['Jaccard Coeff']]  # Jaccard Coefficient as X
y = df6['Actual Class']   # Category as Y

# Split the dataset into training and testing sets (80-20 split)
X3_train, X3_test, y_train, y_test = train_test_split(X3, y, test_size=0.2, random_state=7)

# Initialize the RandomForestClassifier
model3 = RandomForestClassifier()

# Perform 5-fold cross-validation
cv_scores = cross_val_score(model3, X3_train, y_train, cv=5)

# Print the cross-validation scores
print("Cross-validation scores:", cv_scores)
print("Mean CV Score:", np.mean(cv_scores))

# Train the model on the entire training data
model3.fit(X3_train, y_train)

# Evaluate the model on the test set
test_score_Jaccard = model3.score(X3_test, y_test)
print("Test Score for Jaccard coefficient:", test_score_Jaccard)

# Cosine Coeff

X4 = df6[['Cosine Coeff']]  # least vector distance as X
y = df6['Actual Class']   # Category as Y

# Split the dataset into training and testing sets (80-20 split)
X4_train, X4_test, y_train, y_test = train_test_split(X4, y, test_size=0.2, random_state=7)

# Initialize the RandomForestClassifier
model4 = RandomForestClassifier()

# Perform 5-fold cross-validation
cv_scores = cross_val_score(model4, X4_train, y_train, cv=5)

# Print the cross-validation scores
print("Cross-validation scores:", cv_scores)
print("Mean CV Score:", np.mean(cv_scores))

# Train the model on the entire training data
model4.fit(X4_train, y_train)

# Evaluate the model on the test set
test_score_cosine = model4.score(X4_test, y_test)
print("Test Score for Cosine Coefficient:", test_score_cosine)

# Overlap Coeff

X5 = df6[['Overlap Coeff']]  # least vector distance as X
y = df6['Actual Class']   # Category as Y

# Split the dataset into training and testing sets (80-20 split)
X5_train, X5_test, y_train, y_test = train_test_split(X5, y, test_size=0.2, random_state=7)

# Initialize the RandomForestClassifier
model5 = RandomForestClassifier()

# Perform 5-fold cross-validation
cv_scores = cross_val_score(model5, X5_train, y_train, cv=5)

# Print the cross-validation scores
print("Cross-validation scores:", cv_scores)
print("Mean CV Score:", np.mean(cv_scores))

# Train the model on the entire training data
model5.fit(X5_train, y_train)

# Evaluate the model on the test set
test_score_overlap = model5.score(X5_test, y_test)
print("Test Score for Overlap Coefficient:", test_score_overlap)

print("Test Score for Vector Distance:", test_score_vector_distance)
print("Test Score for Dice Coefficient:", test_score_dice)
print("Test Score for Jaccard coefficient:", test_score_Jaccard)
print("Test Score for Cosine Coefficient:", test_score_cosine)
print("Test Score for Overlap Coefficient:", test_score_overlap)

# The best model is the Vector Distance model

predictions = model1.predict(X1)

df_craigslist['Predicted Class'] = predictions

print(df_craigslist)

#print(df_amazon)

values_amazon_pc = []
for i in range(len(df_amazon)):
    values_amazon_pc.append(1)

df_amazon['Predicted Class'] = values_amazon_pc

print(df_amazon)

combined_df = pd.concat([df_craigslist,df_amazon], ignore_index=False)

print(combined_df)



##Part B

####Now let's create Topic modelling and final list of recommendation through sorting
import pandas as pd
# Reading the CSV file with 'ISO-8859-1' encoding
#source_df = pd.read_csv('Dummy Craigslist 1.csv', encoding='ISO-8859-1')
# Filter the DataFrame in-place where 'Classification' equals 1
df = combined_df[combined_df['Predicted Class'] == 1]
# Creating a list of lists from the 'Description' column of the filtered DataFrame
c = [[doc] for doc in df['Description']]
# Print the first 5 entries
print(c[:5])


#Create term-document matrix
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
import string
# Ensure that NLTK's resources are downloaded
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Initialize WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
# Getting a set of English stop words
stop_words = set(stopwords.words('english'))
# Define a function to lemmatize, remove stop words and punctuations
def preprocess(text):
    # Tokenize and lemmatize each word in the text
    lemmatized = [lemmatizer.lemmatize(word) for word in nltk.word_tokenize(text)
                  if word.lower() not in stop_words and word not in string.punctuation]
    return ' '.join(lemmatized)
# Apply the preprocessing to each document
c_processed = [preprocess(' '.join(doc)) for doc in c]
# Using CountVectorizer with min_df and ngram_range parameters
vectorizer = CountVectorizer(min_df=5, ngram_range=(1, 2))
# Fit the model with your preprocessed data
vectorizer.fit(c_processed)
# Print the vocabulary
print(vectorizer.vocabulary_)
# Transform the documents into a document-term matrix
v = vectorizer.transform(c_processed)
# Print the matrix in a dense format
print(v.toarray())
print("Dimensions of the transformed term-document matrix:", v.shape)




#Topic modelling using LDA
from sklearn.decomposition import LatentDirichletAllocation
# Number of topics
n_topics = 10
# Create and fit the LDA model
lda = LatentDirichletAllocation(n_components=n_topics, random_state=0)
lda.fit(v)
# Function to print topics with their top words
def print_topics(model, count_vectorizer, n_top_words):
    words = count_vectorizer.get_feature_names_out()
    for topic_idx, topic in enumerate(model.components_):
        print(f"\nTopic #{topic_idx}:")
        print(" ".join([words[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))


# Number of top words to be displayed per topic
n_top_words = 2
# Print the topics with the top words
print_topics(lda, vectorizer, n_top_words)

# Extracting the topic distribution for each document
doc_topic_dist = lda.transform(v)
# Function to print top topics for each document
def print_top_document_topics(doc_topic_dist, n_top_topics):
    for doc_idx, topics in enumerate(doc_topic_dist):
        top_topics = topics.argsort()[-n_top_topics:][::-1]
        topic_str = " ".join([f"Topic {topic}: {topics[topic]:.2f}" for topic in top_topics])
        print(f"Document #{doc_idx}: {topic_str}")

# Number of top topics to be displayed per document
n_top_topics = 4
# Print the top topics for each document
print_top_document_topics(doc_topic_dist, n_top_topics)


#Labelling topic
import pandas as pd
# Get the most associated topic for each document
most_associated_topics = doc_topic_dist.argmax(axis=1)
# Create a DataFrame
document_topics_df = pd.DataFrame({
    'Document_Index': range(len(most_associated_topics)),
    'Most_Associated_Topic': most_associated_topics
})
# Display the DataFrame
print(document_topics_df)



#Making new Dataframe with topics and original source data
# Reset the index of df to align with document_topics_df
df.reset_index(drop=True, inplace=True)
# Join the df with document_topics_df
# Ensure that both DataFrames have the same number of rows
if len(df) == len(document_topics_df):
    combined_df_2 = df.join(document_topics_df.set_index('Document_Index'))
else:
    print("The number of rows in df and document_topics_df does not match.")
# Display the combined DataFrame
print(combined_df_2)



#Adding topic names
# Extract top words for each topic and store in a dictionary
def get_topic_top_words(model, count_vectorizer, n_top_words):
    topic_top_words = {}
    words = count_vectorizer.get_feature_names_out()
    for topic_idx, topic in enumerate(model.components_):
        top_words = " ".join([words[i] for i in topic.argsort()[:-n_top_words - 1:-1]])
        topic_top_words[topic_idx] = top_words
    return topic_top_words

# Get the dictionary of topic to top words
topic_top_words_dict = get_topic_top_words(lda, vectorizer, n_top_words)
# Map 'Most_Associated_Topic' to its top words and add a 'Topic' column
combined_df_2['Topic'] = combined_df_2['Most_Associated_Topic'].map(topic_top_words_dict)
# Display the updated DataFrame
print(combined_df_2)


#Distance column from the selected city - say 'Indianapolis'
#install geopy in python packages if pycharm or pip install geopy
import pandas as pd
from geopy.distance import geodesic
# Load the dataset (assuming it's a CSV file)
cities_df = pd.read_csv('worldcities.csv')
# Function to get coordinates for a given city
def get_city_coords(city, country, df):
    city_data = df[(df['city_ascii'] == city) & (df['country'] == country)].iloc[0]
    return city_data['lat'], city_data['lng']
# Define the city and country
selected_city = "Indianapolis"
selected_country = "United States"
# Get coordinates for the selected city
selected_city_coords = get_city_coords(selected_city, selected_country, cities_df)
# Function to calculate distance
def calculate_distance(icbm, city_coords):
    try:
        lat, lon = map(float, icbm.split(', '))
        return geodesic(city_coords, (lat, lon)).kilometers
    except:
        return None
# Apply the function to the 'ICBM' column and create a new column based on the selected city
distance_col_name = f"Distance_to_{selected_city.replace(' ', '_')}_km"
combined_df_2[distance_col_name] = combined_df_2['ICBM'].apply(lambda x: calculate_distance(x, selected_city_coords))
# Display the DataFrame
print(combined_df_2)

# Save the combined DataFrame as an Excel file
combined_df_2.to_excel('combined_df_2.xlsx', index=False)
print("DataFrame saved as 'combined_df_2.xlsx'")


#Apply recommendation sort conditions
import pandas as pd
import numpy as np
# Define a custom order for the 'Condition' column
condition_order = ['excellent', 'new', 'likely new', 'good', 'fair']
# Function to generate a sort key for 'Condition'
def condition_sort_key(condition):
    try:
        return condition_order.index(condition)
    except ValueError:
        # Assign a random large number for unspecified conditions
        return np.random.randint(100, 1000)
# Function to determine the column name for distance (example placeholder)
def distance_col_name():
    # Replace this with the actual logic to determine the column name
    return 'Distance_Column'
# Apply the sort key function to the 'Condition' column to create a sortable column
combined_df_2['Condition_Sort'] = combined_df_2['Condition'].apply(condition_sort_key)
# Determine the distance column name
distance_column = f"Distance_to_{selected_city.replace(' ', '_')}_km"
# Group by 'Most_Associated_Topic' and sort within each group
grouped_sorted_dfs = []
for topic, group in combined_df_2.groupby('Most_Associated_Topic'):
    sorted_group = group.sort_values(by=[distance_column, 'Condition_Sort', 'Posted Date'], ascending=[True, True, False])
    grouped_sorted_dfs.append(sorted_group)
# Concatenate the sorted groups back into a single DataFrame
sorted_combined_df_2 = pd.concat(grouped_sorted_dfs)
# Drop the temporary 'Condition_Sort' column
sorted_combined_df_2 = sorted_combined_df_2.drop('Condition_Sort', axis=1)
# Display the updated DataFrame
print(sorted_combined_df_2)
# Save the sorted DataFrame as an Excel file
sorted_combined_df_2.to_excel('recommendations_sorted.xlsx', index=False)
print("DataFrame saved as 'recommendations_sorted.xlsx'")


##Create topic wise recommendations
# Create a new DataFrame by taking the top entry for each 'Topic'
reco_df = sorted_combined_df_2.groupby('Topic').head(1).reset_index(drop=True)
# Select only the 'Topic', 'Product Name', and 'ID' columns
reco_df = reco_df[['Topic', 'Product Name', 'ID']]
# Display the reco_df DataFrame
print(reco_df)

# Save the reco_df DataFrame to an Excel file
reco_df.to_excel("topic_wise_recommendations.xlsx", index=False)
print("Final topic_wise_recommendations file is saved!")

