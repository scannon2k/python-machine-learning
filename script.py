# import necessary packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from IPython.display import display

# sklearn
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

# linear regression
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# scaling
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

# silhouette score
from sklearn.metrics import silhouette_score

# set random seed
RSEED = 50

%matplotlib inline

# to suppress warnings
import warnings
warnings.filterwarnings('ignore')

# load the 'matches' dataset from csv
columns = ['number','pokemon_name','main_type','region','attack','defense','stamina','shiny','shadow']
pokemon = pd.read_csv('pokemon.csv', usecols = columns)

def shape(x):
    rows, cols = x.shape
    print(f"There are {rows} rows and {cols} columns in the dataset")

# observe the rows and columns
shape(pokemon)

# view the dataset with head()
pokemon.head()

pokemon.info()

pokemon.describe()

# check for duplicates
print("Number of duplicates in the pokémon dataset:", pokemon.duplicated().sum())

# check for missing
pokemon.isnull().sum()

# Sort into lists based on datatypes
float_col=[]
object_col=[]
int_col=[]
for col in list(pokemon.columns):
    if pokemon[col].dtype == 'int64':
        int_col.append(col)
    elif pokemon[col].dtype == 'float64':
        float_col.append(col)
    else:
        object_col.append(col)
        
continuous = pokemon[int_col]
objects = pokemon[object_col]

del continuous['number']

for col in continuous.columns[0:]:
    print(col)
    print('Skew :',round(continuous[col].skew(),2))
    plt.figure(figsize=(15,4))
    plt.subplot(1,2,1)
    continuous[col].hist(bins=10, grid=False)
    plt.ylabel('count')
    plt.subplot(1,2,2)
    sns.boxplot(x=continuous[col])
    plt.show()

# shorten the type variable in objects dataframe to 3 letters
objects['main_type'] = objects['main_type'].astype(str).str[0:3]
#del objects['main_type']
objects.head()

sns.set_style("whitegrid")

for col in objects.columns[1:]:
    objects[col].value_counts().plot(kind='bar', rot = 0, alpha=0.8, color='#00429d')
    ax = plt.subplot(111)
    ax.spines[['top', 'right','left','bottom']].set_visible(True)
    #ax.grid(False)
    ax.grid(axis='x')
    ax.set_facecolor("white")
    plt.show()

# pairplot
sns.pairplot(continuous, corner=True)

# plot seaborn heatmap
plt.figure(figsize=(8,8))
sns.heatmap(continuous.corr(), annot=True, cmap='Blues').set_title('Correlation Matrix') #cmap='viridis'

#Scaling the data and storing the output as a new dataframe
scaler=StandardScaler()
data_scaled=pd.DataFrame(scaler.fit_transform(continuous), columns=continuous.columns)

print(data_scaled.head())

#Creating copy of the data to store labels from each algorithm
data_scaled_copy = data_scaled.copy(deep=True)

#Empty dictionary to store the SSE for each value of k
sse = {} 

# iterate for a range of Ks and fit the scaled data to the algorithm. Use inertia attribute from the clustering object and 
# store the inertia value for that k 
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, random_state=1).fit(data_scaled)
    sse[k] = kmeans.inertia_

#Elbow plot
plt.figure()
plt.plot(list(sse.keys()), list(sse.values()), 'bx-')
plt.xlabel("Number of cluster")
plt.ylabel("SSE")
plt.show()

#Empty dictionary to store the Silhouette score for each value of k
sc = {} 

# iterate for a range of Ks and fit the scaled data to the algorithm. Store the Silhouette score for that k 
for k in range(2, 10):
    kmeans = KMeans(n_clusters=k, random_state=1).fit(data_scaled)
    labels = kmeans.predict(data_scaled)
    sc[k] = silhouette_score(data_scaled, labels)

#Elbow plot
plt.figure()
plt.plot(list(sc.keys()), list(sc.values()), 'bx-')
plt.xlabel("Number of cluster")
plt.ylabel("Silhouette Score")
plt.show()

kmeans = KMeans(n_clusters=3, random_state=1)
kmeans.fit(data_scaled)

#Adding predicted labels to the original data and scaled data 
data_scaled_copy['KMeans_Labels'] = kmeans.predict(data_scaled)
continuous['KMeans_Labels'] = kmeans.predict(data_scaled)

continuous['KMeans_Labels'].value_counts()

#Calculating mean and median of the original data for each label
mean = continuous.groupby('KMeans_Labels').mean()
median = continuous.groupby('KMeans_Labels').median()
df_kmeans = pd.concat([mean, median], axis=0)
df_kmeans.index = ['group_0 Mean', 'group_1 Mean', 'group_2 Mean', 'group_0 Median', 'group_1 Median', 'group_2 Median']
df_kmeans.T

almonds = ['attack','defense','stamina']

for col in almonds:
    sns.boxplot(x = 'KMeans_Labels', y = col, data = continuous)
    plt.show()

walnuts = ['attack','defense']

for col in walnuts:
    sns.scatterplot(x = col, y = 'stamina',data = continuous ,hue='KMeans_Labels', palette='Dark2')
    plt.show()

# load the test and train datasets: 70-30 split
train_df = pd.read_excel('poktrain.xlsx')
test_df = pd.read_excel('poktest.xlsx')

train_df = train_df.drop(['number', 'pokemon_name','main_type','region'], axis=1)
test_df = test_df.drop(['number', 'pokemon_name','main_type','region'], axis=1)

pd.get_dummies(train_df)

train_df.head()

test_df.head()

fig, axes = plt.subplots(1, 3, figsize=(20, 6))
  
fig.suptitle('Histogram for all numerical variables in the dataset')
  
sns.histplot(x='attack', data=train_df, kde=True, ax=axes[0]);
sns.histplot(x='defense', data=train_df, kde=True, ax=axes[1]);
sns.histplot(x='stamina', data=train_df, kde=True, ax=axes[2]);

# we are removing the outcome variable from the feature set and also the variable Outlet_Establishment_Year as we have created
# a new variable Outlet_Age
train_features = train_df.drop(['shiny'], axis=1)

# and then we are extracting the outcome variable separately
train_target = train_df['shiny']
train_target = pd.get_dummies(train_target, drop_first=True)
train_target.rename(columns = {'Yes':'shiny'}, inplace = True)

# in linear based models it is mandatory to create dummy variables for the categorical variables
train_features = pd.get_dummies(train_features, drop_first=True)
train_features.head()

# creating an instance of the MinMaxScaler
scaler = MinMaxScaler()

# applying fit_transform on the training features data
train_features_scaled = scaler.fit_transform(train_features)
#test_features_scaled = scaler.transform(test_features)


# the above scaler returns the data in array format, below we are converting back to pandas dataframe
train_features_scaled = pd.DataFrame(train_features_scaled, index=train_features.index, columns=train_features.columns)
train_features_scaled.head()

# here we are adding the intercept term
train_features_scaled = sm.add_constant(train_features_scaled)

# calling the OLS algorithm on the train features and target variable
ols_model_0 = sm.OLS(train_target, train_features_scaled)

# fitting the model
ols_res_0 = ols_model_0.fit()

print(ols_res_0.summary())

df = pd.read_csv('pokemon.csv').sample(770, random_state = RSEED)
#df.drop(['number','pokemon_name'], axis=1)
del df['number']
del df['pokemon_name']
# must turn categorical variables into dummies
df = pd.get_dummies(df, drop_first=True)
df.head()

df = df.select_dtypes('number')

df['shiny_Yes'] = df['shiny_Yes'].replace({2: 0})
df = df.loc[df['shiny_Yes'].isin([0, 1])].copy()
df = df.rename(columns = {'shiny_Yes': 'label'})
df['label']=1-df['label']
df['label'].value_counts()

from sklearn.model_selection import train_test_split

# Extract the labels
labels = np.array(df.pop('label'))

# 30% examples in test data
train, test, train_labels, test_labels = train_test_split(df, labels, 
                                                          stratify = labels,
                                                          test_size = 0.3, 
                                                          random_state = RSEED)

train = train.fillna(train.mean())
test = test.fillna(test.mean())

# Features for feature importances
features = list(train.columns)

train.shape

test.shape

# Train tree
tree = DecisionTreeClassifier(random_state=RSEED)

tree.fit(train, train_labels)
print(f'Decision tree has {tree.tree_.node_count} nodes with maximum depth {tree.tree_.max_depth}.')

print(tree.predict_proba(train))

# Make probability predictions
train_probs = tree.predict_proba(train)[:, 1]
probs = tree.predict_proba(test)[:, 1]

train_predictions = tree.predict(train)
predictions = tree.predict(test)

from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve

print(f'Train ROC AUC Score: {roc_auc_score(train_labels, train_probs)}')
print(f'Test ROC AUC  Score: {roc_auc_score(test_labels, probs)}')

from sklearn.metrics import confusion_matrix
import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Oranges):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Source: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.figure(figsize = (10, 10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, size = 24)
    plt.colorbar(aspect=4)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, size = 14)
    plt.yticks(tick_marks, classes, size = 14)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    
    # Labeling the plot
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), fontsize = 20,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
        
    plt.grid(None)
    plt.tight_layout()
    plt.ylabel('True label', size = 18)
    plt.xlabel('Predicted label', size = 18)

cm = confusion_matrix(test_labels, predictions)
plot_confusion_matrix(cm, classes = [ 'Shiny', 'Not Shiny'],
                      title = 'Pokemon Confusion Matrix',cmap=plt.cm.Blues)

fi = pd.DataFrame({'feature': features,
                   'importance': tree.feature_importances_}).\
                    sort_values('importance', ascending = False)
fi.head()

import subprocess
from sklearn.tree import DecisionTreeClassifier, export_graphviz, plot_tree
from PIL import Image
#from subprocess import check_call
# Save tree as dot file
#export_graphviz(tree, 'tree_real_data.dot', rounded = True, 
#                feature_names = features, max_depth = 6,
#                class_names = ['poor health', 'good health'], filled = True)

# Convert to png
#subprocess.call(['dot', '-Tpng', 'tree_real_data.dot', '-o', 'tree_real_data.png', '-Gdpi=200'])

#tree.export_graphviz(model_name,'m1_ent.dot',filled=True,feature_names=features,
#                     class_names= classes)

#convert dot into image
#subprocess.check_call(['dot','-Tpng','tree_real_data.dot','-0','tree_real_data.png'])

# Visualize
#Image(filename='tree_real_data.png')
fig = plt.figure(figsize=(25,20))
plot_tree(tree, max_depth=3, feature_names = features)

from sklearn.ensemble import RandomForestClassifier

# Create the model with 100 trees
model = RandomForestClassifier(n_estimators=100, 
                               random_state=RSEED, 
                               max_features = 'sqrt',
                               n_jobs=-1, verbose = 1,
                              class_weight='balanced')

# Fit on training data
model.fit(train, train_labels)

n_nodes = []
max_depths = []

for ind_tree in model.estimators_:
    n_nodes.append(ind_tree.tree_.node_count)
    max_depths.append(ind_tree.tree_.max_depth)
    
print(f'Average number of nodes {int(np.mean(n_nodes))}')
print(f'Average maximum depth {int(np.mean(max_depths))}')

train_rf_predictions = model.predict(train)
train_rf_probs = model.predict_proba(train)[:, 1]

rf_predictions = model.predict(test)
rf_probs = model.predict_proba(test)[:, 1]

evaluate_model(rf_predictions, rf_probs, train_rf_predictions, train_rf_probs)

cm = confusion_matrix(test_labels, rf_predictions)
plot_confusion_matrix(cm, classes = [ 'Shiny', 'Not Shiny'],
                      title = 'Pokemon Confusion Matrix',cmap=plt.cm.Blues)

fi_model = pd.DataFrame({'feature': features,
                   'importance': model.feature_importances_}).\
                    sort_values('importance', ascending = False)
fi_model.head(10)
