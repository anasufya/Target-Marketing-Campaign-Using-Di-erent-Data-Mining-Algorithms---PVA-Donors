# 1. LOAD LIBRARIES
import numpy as np
import scipy as sc
import pandas as pd
import matplotlib.pyplot as plt
from pandas_profiling import ProfileReport
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from pandas_profiling import ProfileReport
import pandas_profiling as pp
import seaborn as sns
from datetime import datetime
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.neighbors import KNeighborsClassifier
from sompy.sompy import SOMFactory
from sompy.visualization.mapview import View2D
from sompy.visualization.bmuhits import BmuHitsView
from sompy.visualization.hitmap import HitMapView
from kmodes.kmodes import KModes
from scipy.cluster.hierarchy import dendrogram, linkage
import scipy.cluster.hierarchy as shc
from matplotlib import pyplot
from sklearn.cluster import AgglomerativeClustering
import plotly.express as pe
from plotly.offline import plot

# 2. IMPORT DATASET
data = pd.read_csv("donors.csv", header=0, sep=',',na_values=' ')

# 3. FEATURE SELECTION
# 3.1 Select the important features according
data = data[['INCOME','TIMELAG','MINRAMNT','HIT','AVGGIFT','CARDPROM','PVASTATE','DOMAIN',
          'MALEVET','LOCALGOV','STATEGOV','FEDGOV','STATEGOV','HOMEOWNR',
          'WEALTH2','GENDER','MAJOR','VETERANS','PEPSTRFL','ETH7','ETH10','ETH11','AFC1','AFC2',
          'AFC3','AFC4','AFC5','AFC6','VC1','VC2','VC3','VC4','NUMPRM12','PCOWNERS','LASTGIFT',
          'MDMAUD_R','MDMAUD_F','MDMAUD_A','MDMAUD','LASTDATE', 'NUMPROM','MAXADATE', 'DOB']]

# 4. METADATA
metadata={'MALEVET':'Males Veterans','lOCALGOV':'Employed by Local Gov','STATEGOV':'Employed by State Gov',
          'FEDGOV':'Employed by Fed Gov','AFC4':'% Adult Veterans Age 16+','AFC5':'% Male Veterans Age 16+',
          'AFC6':'% Female Veterans Age 16+','VC1':'% Vietnam Veterans Age 16+','VC2':'% Korean Veterans Age 16+',
          'VC3':'% WW2 Veterans Age 16+','VC4':'% Veterans Serving After May 1995 Only','AGE':'Age','HOMEOWNR':'Homeowner',
          'PEPSTRFL':'PEP Star RFA Status','MALE':'Male','OTHER':'Other than Male','DOMAIN_U':'Urban Area','DOMAIN_S':'Suburban Area',
          'DOMAIN_T':'Town Area','DOMAIN_R':'Rural Area','INCOME':'Income','AVGGIFT':'Average Dollar Gift',
          'CARDPROM':'Card Promotion Received to Date','NUMPRM12':'Promotions Received in the Last 12 Months','LASTGIFT':'Value of Most Recent Gift',
          'MAJOR':'Major Donor','WEALTH2':'Wealth Rating Using Median Family Income','MDMAUD_R_C':'Current Donor','MDMAUD_R_L':'Lapsed Donor',
          'MDMAUD_R_D':'Dormant Donor','MDMAUD_R_I':'Inactive Donor','MDMAUD_F_1':'One Gift','MDMAUD_F_2':'Two-Four Gifts',
          'MDMAUD_F_3':'More Than 4 Gifts','MDMAUD_A_L':'Giving Less Than 100$','MDMAUD_A_M':'Giving 500 to 999$','MDMAUD_A_T':'Giving More Than 1000$'}



# 5. MISSING VALUES TREATMENT
# Replace empty spaces to nan (Copy df to data)
data = data.replace(r'^\s*$',np.NaN, regex=True)
# Copy data to df dataset
df = data.copy()

# 5.1 Check missings
# 5.1.1 By column
# Check the total missing values
df.isnull().sum()
# Percentage of missing values in each column
nan_percentage = df.isna().sum()/len(df)*100.00
# Total value of missing values
df.isna().values.sum()
# Select the columns with a "nan" percentage above 40%
above_na = nan_percentage[nan_percentage>=40]
#print('Número de variaveis a eliminar: ', above_na)
# Features 'PVSTATE', 'VETERANS' and 'PCOWNRS' will be dropped
df = df.drop(columns=['VETERANS', 'PVASTATE', 'PCOWNERS'])
df = df.reset_index()

# 5.1.2 By row 
missings_row = pd.DataFrame(df.isnull().sum(axis=1)).reset_index()


# 6. Create new variable 'Age' from 'DOB'
# Cast of DOB to a datetime object
df['DOB']= pd.to_datetime(df['DOB'])
year = []
for date in df['DOB']:
   year.append(date.year)
year = pd.Series(year)
df['AGE'] = year.astype(int,errors='ignore')
df['AGE'] = 2020-df['AGE']
# Drop the column 'DOB' because its irrelevant
df = df.drop(columns=['DOB'])

# 7 Fill missing values 
# 7.1 Fill missing values with 0
df['MAJOR'] = df['MAJOR'].fillna(0)
df['PEPSTRFL'] = df['PEPSTRFL'].fillna(0)

# 7.2 Fill missing values with median - Because of outliers (more robust)
df['INCOME'] = df['INCOME'].fillna(df['INCOME'].mean())
df['TIMELAG'] = df['TIMELAG'].fillna(df['TIMELAG'].median())
df['AGE'] = df['AGE'].fillna(df['AGE'].median())

# 7.3 Fill missing values with mode (Binary features)
df['WEALTH2'] = df['WEALTH2'].fillna(df['WEALTH2'].mode().loc[0])
df['HOMEOWNR'] = df['HOMEOWNR'].fillna(df['HOMEOWNR'].mode().loc[0])
df['GENDER'] = df['GENDER'].fillna(df['GENDER'].mode().loc[0])
df['DOMAIN'] = df['DOMAIN'].fillna(df['DOMAIN'].mode().loc[0])

# 8. DEAL WITH BINARY/CATEGORICAL FEATURES
categorical_features = df.select_dtypes(include=['object'])
# 8.1 Add 'WEALTH2' column in the categorical features
categorical_features['WEALTH2'] = df['WEALTH2']
# 8.2 Replace'HOMEOWNR' categories for binary 
categorical_features['HOMEOWNR'] = categorical_features['HOMEOWNR'].replace('H',1)
categorical_features['HOMEOWNR'] = categorical_features['HOMEOWNR'].replace('U',0)
# 8.3 Replace 'MAJOR' categories for binary
categorical_features['MAJOR'] = categorical_features['MAJOR'].replace('X',1)
# 8.4 Replace 'PEPSTRFL' categories for binary
categorical_features['PEPSTRFL'] = categorical_features['PEPSTRFL'].replace('X',1)
# 8.4 Create a separate value as ‘Other’ - U
categorical_features['GENDER'] = categorical_features['GENDER'].replace(['C','J','U', 'A'],'O')


# 9. DEAL WITH DATE TIME FEATURES 
# 9.1 Convert 'LASTDATE' and 'MAXADATE' to a datetime series
categorical_features['LASTDATE']= pd.to_datetime(categorical_features['LASTDATE'])
categorical_features['MAXADATE']= pd.to_datetime(categorical_features['MAXADATE'])

# 10. DEAL WITH NON-CATEGORICAL FEATURES
non_categorical = df.copy()

non_categorical = non_categorical.drop(columns = categorical_features.columns)
# 10.1 'WEALTH2' is a categorical feature (rating)
#non_categorical = non_categorical.drop(columns = 'WEALTH2')

# 10.2 Drop the duplicate columns
non_categorical = non_categorical.T.drop_duplicates().T

# 10.3 Drop the 'index' columns
df = df.drop(columns='index')
# 10.4 Reset the index in non_categorical data
non_categorical = non_categorical.reset_index()

non_categorical = non_categorical.drop(columns='index')

# 11. OUTLIER ANALYSIS
# 11.1 Univariate Analysis

# 11.1.1 Create a new columns 'Outlier'
non_categorical['Outlier_Uni'] = 0
'''
# 11.1.2 Plot the boxplots for each feature
f, axes = plt.subplots(6, 5, figsize=(20, 20))  
plt.subplots_adjust(wspace = 0.3, hspace = 0.3)  
sns.boxplot(non_categorical["INCOME"], color=sns.color_palette("Blues")[1], ax=axes[0, 0])
sns.boxplot(non_categorical["TIMELAG"], color=sns.color_palette("Blues")[3], ax=axes[0, 1])
sns.boxplot(non_categorical["MINRAMNT"], whis=5, color=sns.color_palette("Blues")[5], ax=axes[0, 2])
sns.boxplot(non_categorical["HIT"], color=sns.color_palette("BuGn_r")[4], ax=axes[0, 3])
sns.boxplot(non_categorical["AVGGIFT"], color=sns.color_palette("BuGn_r")[3], ax=axes[0, 4])
sns.boxplot(non_categorical["CARDPROM"], whis=7, color=sns.color_palette("BuGn_r")[0], ax=axes[1, 0])
sns.boxplot(non_categorical["MALEVET"], whis=2.5, color=sns.cubehelix_palette(8)[2], ax=axes[1, 1])
sns.boxplot(non_categorical["LOCALGOV"], whis=7.5, color=sns.cubehelix_palette(8)[4], ax=axes[1, 2])
sns.boxplot(non_categorical["FEDGOV"], whis=7, color=sns.cubehelix_palette(8)[6], ax=axes[1, 3])
sns.boxplot(non_categorical["ETH7"]>0, whis=7, color=sns.cubehelix_palette(8)[6], ax=axes[1, 4])
sns.boxplot(non_categorical["ETH10"]>0, whis=7, color=sns.cubehelix_palette(8)[6], ax=axes[2, 0])
sns.boxplot(non_categorical["ETH11"], whis=7, color=sns.cubehelix_palette(8)[6], ax=axes[2, 1])
sns.boxplot(non_categorical["AFC1"], whis=7, color=sns.cubehelix_palette(8)[6], ax=axes[2, 2])
sns.boxplot(non_categorical["AFC2"], whis=7, color=sns.cubehelix_palette(8)[6], ax=axes[2, 3])
sns.boxplot(non_categorical["AFC3"], whis=7, color=sns.cubehelix_palette(8)[6], ax=axes[2, 4])
sns.boxplot(non_categorical["AFC4"], whis=7, color=sns.cubehelix_palette(8)[6], ax=axes[3, 0])
sns.boxplot(non_categorical["AFC5"], whis=7, color=sns.cubehelix_palette(8)[6], ax=axes[3, 1])
sns.boxplot(non_categorical["AFC6"], whis=7, color=sns.cubehelix_palette(8)[6], ax=axes[3, 2])
sns.boxplot(non_categorical["VC1"], whis=7, color=sns.cubehelix_palette(8)[6], ax=axes[3, 3])
sns.boxplot(non_categorical["VC2"], whis=7, color=sns.cubehelix_palette(8)[6], ax=axes[3, 4])
sns.boxplot(non_categorical["VC3"], whis=7, color=sns.cubehelix_palette(8)[6], ax=axes[4, 0])
sns.boxplot(non_categorical["VC4"], whis=7, color=sns.cubehelix_palette(8)[6], ax=axes[4, 1])
sns.boxplot(non_categorical["NUMPRM12"], whis=7, color=sns.cubehelix_palette(8)[6], ax=axes[4, 2])
sns.boxplot(non_categorical["LASTGIFT"], whis=7, color=sns.cubehelix_palette(8)[6], ax=axes[4, 3])
sns.boxplot(non_categorical["NUMPROM"], whis=7, color=sns.cubehelix_palette(8)[6], ax=axes[4, 4])
sns.boxplot(non_categorical["AGE"], whis=7, color=sns.cubehelix_palette(8)[6], ax=axes[5, 0])

# 11.1.3 Histogram visualization
f, axes = plt.subplots(6, 5, figsize=(20, 20))  
plt.subplots_adjust(wspace = 0.3, hspace = 0.3)  
sns.distplot(non_categorical["INCOME"], color=sns.color_palette("Blues")[1], ax=axes[0, 0])
sns.distplot(non_categorical["TIMELAG"], color=sns.color_palette("Blues")[3], ax=axes[0, 1])
sns.distplot(non_categorical["MINRAMNT"], color=sns.color_palette("Blues")[5], ax=axes[0, 2])
sns.distplot(non_categorical["HIT"], color=sns.color_palette("BuGn_r")[4], ax=axes[0, 3])
sns.distplot(non_categorical["AVGGIFT"], color=sns.color_palette("BuGn_r")[3], ax=axes[0, 4])
sns.distplot(non_categorical["CARDPROM"], color=sns.color_palette("BuGn_r")[0], ax=axes[1, 0])
sns.distplot(non_categorical["MALEVET"], color=sns.cubehelix_palette(8)[2], ax=axes[1, 1])
sns.distplot(non_categorical["LOCALGOV"], color=sns.cubehelix_palette(8)[4], ax=axes[1, 2])
sns.distplot(non_categorical["FEDGOV"],color=sns.cubehelix_palette(8)[6], ax=axes[1, 3])
sns.distplot(non_categorical["ETH7"]>0, color=sns.cubehelix_palette(8)[6], ax=axes[1, 4])
sns.distplot(non_categorical["ETH10"]>0, color=sns.cubehelix_palette(8)[6], ax=axes[2, 0])
sns.distplot(non_categorical["ETH11"], color=sns.cubehelix_palette(8)[6], ax=axes[2, 1])
sns.distplot(non_categorical["AFC1"], color=sns.cubehelix_palette(8)[6], ax=axes[2, 2])
sns.distplot(non_categorical["AFC2"], color=sns.cubehelix_palette(8)[6], ax=axes[2, 3])
sns.distplot(non_categorical["AFC3"], color=sns.cubehelix_palette(8)[6], ax=axes[2, 4])
sns.distplot(non_categorical["AFC4"], color=sns.cubehelix_palette(8)[6], ax=axes[3, 0])
sns.distplot(non_categorical["AFC5"], color=sns.cubehelix_palette(8)[6], ax=axes[3, 1])
sns.distplot(non_categorical["AFC6"], color=sns.cubehelix_palette(8)[6], ax=axes[3, 2])
sns.distplot(non_categorical["VC1"], color=sns.cubehelix_palette(8)[6], ax=axes[3, 3])
sns.distplot(non_categorical["VC2"], color=sns.cubehelix_palette(8)[6], ax=axes[3, 4])
sns.distplot(non_categorical["VC3"], color=sns.cubehelix_palette(8)[6], ax=axes[4, 0])
sns.distplot(non_categorical["VC4"], color=sns.cubehelix_palette(8)[6], ax=axes[4, 1])
sns.distplot(non_categorical["NUMPRM12"], color=sns.cubehelix_palette(8)[6], ax=axes[4, 2])
sns.distplot(non_categorical["LASTGIFT"], color=sns.cubehelix_palette(8)[6], ax=axes[4, 3])
sns.distplot(non_categorical["NUMPROM"], color=sns.cubehelix_palette(8)[6], ax=axes[4, 4])
sns.distplot(non_categorical["AGE"], color=sns.cubehelix_palette(8)[6], ax=axes[5, 0])
'''
# 11.1.4  the features 'AFC1', 'AFC2', 'AFC3', 'EHT11', 'ETH10', 'ETH11'
non_categorical = non_categorical.drop(columns = ['AFC1', 'AFC2', 'AFC3', 'ETH11', 'ETH10', 'ETH11'])
df = df.drop(columns = ['AFC1', 'AFC2', 'AFC3', 'ETH11', 'ETH10', 'ETH11'])

# 11.1.5 Check the number of outliers and locate them in 'Outiler' = 1
# 'TIMELAG'
timelag_mean = non_categorical['TIMELAG'].mean()
timelag_std = non_categorical['TIMELAG'].std()
timelag_outlier = non_categorical.loc[non_categorical['TIMELAG'] > timelag_mean + 5 * timelag_std]['TIMELAG']
timelag_outlier_index = timelag_outlier.index
non_categorical.loc[timelag_outlier_index, 'Outlier_Uni'] = 1
# 'MINRAMNT'
minramnt_mean = non_categorical['MINRAMNT'].mean()
minramnt_std = non_categorical['MINRAMNT'].std()
minramnt_outlier = non_categorical.loc[non_categorical['TIMELAG'] > minramnt_mean + 5 * minramnt_std]['TIMELAG']
minramnt_outlier_index = minramnt_outlier.index
non_categorical.loc[minramnt_outlier_index, 'Outlier_Uni'] = 1
# 'HIT'
hit_mean = non_categorical['HIT'].mean()
hit_std = non_categorical['HIT'].std()
hit_outlier = non_categorical.loc[non_categorical['HIT'] > hit_mean + 5 * hit_std]['TIMELAG']
hit_outlier_index = hit_outlier.index
non_categorical.loc[hit_outlier_index, 'Outlier_Uni'] = 1
# 'AVGGIFT'
avggift_mean = non_categorical['AVGGIFT'].mean()
avggift_std = non_categorical['AVGGIFT'].std()
avggift_outlier = non_categorical.loc[non_categorical['AVGGIFT'] > avggift_mean + 5 * avggift_std]['AVGGIFT']
avggift_outlier_index = avggift_outlier.index
non_categorical.loc[avggift_outlier_index, 'Outlier_Uni'] = 1
# 'MALEVET'
malevet_mean = non_categorical['MALEVET'].mean()
malevet_std = non_categorical['MALEVET'].std()
malevet_outlier = non_categorical.loc[non_categorical['MALEVET'] > malevet_mean + 5 * malevet_std]['AVGGIFT']
malevet_outlier_index = malevet_outlier.index
non_categorical.loc[malevet_outlier_index, 'Outlier_Uni'] = 1
# 'LOCALGOV'
localgov_mean = non_categorical['LOCALGOV'].mean()
localgov_std = non_categorical['LOCALGOV'].std()
localgov_outlier = non_categorical.loc[non_categorical['LOCALGOV'] > localgov_mean + 5 * localgov_std]['AVGGIFT']
localgov_outlier_index = localgov_outlier.index
non_categorical.loc[localgov_outlier_index, 'Outlier_Uni'] = 1
# 'FEDGOV'
fedgov_mean = non_categorical['FEDGOV'].mean()
fedgov_std = non_categorical['FEDGOV'].std()
fedgov_outlier = non_categorical.loc[non_categorical['FEDGOV'] > fedgov_mean + 5 * fedgov_std]['FEDGOV']
fedgov_outlier_index = fedgov_outlier.index
non_categorical.loc[fedgov_outlier_index, 'Outlier_Uni'] = 1
# 'AFC4'
afc4_mean = non_categorical['AFC4'].mean()
afc4_std = non_categorical['AFC4'].std()
afc4_outlier = non_categorical.loc[non_categorical['AFC4'] > afc4_mean + 5 * afc4_std]['AFC4']
afc4_outlier_index = afc4_outlier.index
non_categorical.loc[afc4_outlier_index, 'Outlier_Uni'] = 1
# 'AFC6'
afc6_mean = non_categorical['AFC6'].mean()
afc6_std = non_categorical['AFC6'].std()
afc6_outlier = non_categorical.loc[non_categorical['AFC6'] > afc6_mean + 5 * afc6_std]['AFC6']
afc6_outlier_index = afc6_outlier.index
non_categorical.loc[afc6_outlier_index, 'Outlier_Uni'] = 1
# 'NUMPRM12'
numprm12_mean = non_categorical['NUMPRM12'].mean()
numprm12_std = non_categorical['NUMPRM12'].std()
numprm12_outlier = non_categorical.loc[non_categorical['NUMPRM12'] > numprm12_mean + 5 * numprm12_std]['NUMPRM12']
numprm12_outlier_index = numprm12_outlier.index
non_categorical.loc[numprm12_outlier_index, 'Outlier_Uni'] = 1
# 'LASTGIFT'
lastgift_mean = non_categorical['LASTGIFT'].mean()
lastgift_std = non_categorical['LASTGIFT'].std()
lastgift_outlier = non_categorical.loc[non_categorical['LASTGIFT'] > lastgift_mean + 5 * lastgift_std]['LASTGIFT']
lastgift_outlier_index = lastgift_outlier.index
non_categorical.loc[lastgift_outlier_index, 'Outlier_Uni'] = 1


# 11.2 Multivariate Analysis
# 11.2.1 Numerical features Standardization
std_data = stats.zscore(non_categorical)
# 11.2.2 Initialize K-Means with 70 clusters
k_means = KMeans(n_clusters = 70, init = 'k-means++', n_init =50, max_iter = 300).fit(std_data)
# 11.2.3 Clusters table has the cluster that each observation belongs to ('Cluster'), the original ID of the variables ('ID').
clusters = pd.DataFrame(k_means.labels_, columns=['Cluster'])
# 11.2.4 Associate a column 'N'= n times a clusters repeats
clusters['N'] = clusters['Cluster'].map(clusters['Cluster'].value_counts())
# 11.2.5 Merge the columns in the dataframe
non_categorical = pd.merge(clusters, non_categorical, on=non_categorical.index, how='inner')
# 11.2.6 Filter the outliers <=14
n_filter = non_categorical.loc[non_categorical['N']<=200].index
# 11.2.7 Create the column 'Outlier_Multi' to associate with the outliers
non_categorical['Outlier_Multi'] = 0
non_categorical.loc[n_filter, 'Outlier_Multi'] = 1
# 11.2.8 Final dataset with Outlier_Uni,Outlier_Multi = 0
non_categorical = non_categorical.loc[(non_categorical['Outlier_Uni'] == 0) & (non_categorical['Outlier_Multi'] == 0)]


# 8. TRANSFORM CATEGORICAL/ORDINAL FEATURES

# Reset Index of categorical_features
categorical_features = categorical_features.reset_index()

# 'MDMAUD_R' - Create each column for recency of given: C, L, I, D , 1 and 0 - Non Major donor
categorical_features['MDMAUD_R_C'] = 0
categorical_features['MDMAUD_R_L'] = 0
categorical_features['MDMAUD_R_D'] = 0
categorical_features['MDMAUD_R_I'] = 0
categorical_features.loc[categorical_features['MDMAUD_R'] == 'C', 'MDMAUD_R_C'] = 3
categorical_features.loc[categorical_features['MDMAUD_R'] == 'L', 'MDMAUD_R_L'] = 2
categorical_features.loc[categorical_features['MDMAUD_R'] == 'D', 'MDMAUD_R_D'] = 1
categorical_features.loc[categorical_features['MDMAUD_R'] == 'I', 'MDMAUD_R_I'] = 1
# Drop column 'MDMAUD_R'
categorical_features = categorical_features.drop(columns=['MDMAUD_R'])

# 'MDMAUD_F' - Create each column for frequency code: 1,2,5 , 1 and 0 - Non Major donor
categorical_features['MDMAUD_F_1'] = 0
categorical_features['MDMAUD_F_2'] = 0
categorical_features['MDMAUD_F_5'] = 0
categorical_features.loc[categorical_features['MDMAUD_F'] == '1', 'MDMAUD_F_1'] = 1
categorical_features.loc[categorical_features['MDMAUD_F'] == '2', 'MDMAUD_F_2'] = 2
categorical_features.loc[categorical_features['MDMAUD_F'] == '5', 'MDMAUD_F_5'] = 3
# Drop column 'MDMAUD_F'
categorical_features = categorical_features.drop(columns=['MDMAUD_F'])

# 'MDMAUD_A' - Create each column for donation amount: C, L, M, T , 1 and 0 - Non Major donor
categorical_features['MDMAUD_A_C'] = 0
categorical_features['MDMAUD_A_L'] = 0
categorical_features['MDMAUD_A_M'] = 0
categorical_features['MDMAUD_A_T'] = 0
categorical_features.loc[categorical_features['MDMAUD_A'] == 'C', 'MDMAUD_A_C'] = 2
categorical_features.loc[categorical_features['MDMAUD_A'] == 'L', 'MDMAUD_A_L'] = 1
categorical_features.loc[categorical_features['MDMAUD_A'] == 'M', 'MDMAUD_A_M'] = 3
categorical_features.loc[categorical_features['MDMAUD_A'] == 'T', 'MDMAUD_A_T'] = 4

# 'MDMAUD' will be dropped
categotical_features = categorical_features.drop(columns = ['MDMAUD'])

# 'GENDER' - Create 2 new columns,  'MALE': O or 1 and 'OTHER': 0 or 1
categorical_features['MALE'] = 0
categorical_features['OTHER'] = 0
categorical_features.loc[categorical_features['GENDER'] == 'F', 'MALE'] = 0
categorical_features.loc[categorical_features['GENDER'] == 'M', 'MALE'] = 1
categorical_features.loc[categorical_features['GENDER'] == 'O', 'OTHER'] = 1
# Delete the column 'GENDER'
categorical_features = categorical_features.drop(columns = ['GENDER'])

# 'DOMAIN' will be reduced to only the first byte (Urbanicity level) - U, C, S, T, R
categorical_features['DOMAIN_U'] = 0
categorical_features['DOMAIN_S'] = 0
categorical_features['DOMAIN_T'] = 0
categorical_features['DOMAIN_R'] = 0
categorical_features['DOMAIN_C'] = 0
categorical_features.loc[categorical_features['DOMAIN'] == 'U1', 'DOMAIN_U'] = 1
categorical_features.loc[categorical_features['DOMAIN'] == 'U2', 'DOMAIN_U'] = 1
categorical_features.loc[categorical_features['DOMAIN'] == 'U3', 'DOMAIN_U'] = 1
categorical_features.loc[categorical_features['DOMAIN'] == 'U4', 'DOMAIN_U'] = 1
categorical_features.loc[categorical_features['DOMAIN'] == 'R1', 'DOMAIN_R'] = 1
categorical_features.loc[categorical_features['DOMAIN'] == 'R2', 'DOMAIN_R'] = 1
categorical_features.loc[categorical_features['DOMAIN'] == 'R3', 'DOMAIN_R'] = 1
categorical_features.loc[categorical_features['DOMAIN'] == 'S1', 'DOMAIN_S'] = 1
categorical_features.loc[categorical_features['DOMAIN'] == 'S2', 'DOMAIN_S'] = 1
categorical_features.loc[categorical_features['DOMAIN'] == 'S3', 'DOMAIN_S'] = 1
categorical_features.loc[categorical_features['DOMAIN'] == 'T1', 'DOMAIN_T'] = 1
categorical_features.loc[categorical_features['DOMAIN'] == 'T2', 'DOMAIN_T'] = 1
categorical_features.loc[categorical_features['DOMAIN'] == 'T3', 'DOMAIN_T'] = 1
categorical_features.loc[categorical_features['DOMAIN'] == 'R1', 'DOMAIN_R'] = 1
categorical_features.loc[categorical_features['DOMAIN'] == 'R2', 'DOMAIN_R'] = 1
categorical_features.loc[categorical_features['DOMAIN'] == 'R3', 'DOMAIN_R'] = 1
categorical_features.loc[categorical_features['DOMAIN'] == 'C1', 'DOMAIN_C'] = 1
categorical_features.loc[categorical_features['DOMAIN'] == 'C2', 'DOMAIN_C'] = 1
categorical_features.loc[categorical_features['DOMAIN'] == 'C3', 'DOMAIN_C'] = 1



# Delete the column 'DOMAIN'
categorical_features = categorical_features.drop(columns=['DOMAIN'])

# 12. FINAL DATAFRAME - CONCAT NON-CATEGORICAL AND CATEGORICAL DATAFRAMES (DF)
df = pd.concat([non_categorical, categorical_features], axis=1, sort=False, join='inner')
# 12.1 Drop of unnecessary features
df = df.drop(columns = ['key_0', 'Cluster', 'N', 'level_0', 'Outlier_Uni', 'Outlier_Multi', 'index'])
# 12.1 Reset index of df
df = df.reset_index()
df = df.drop(columns = 'index')
# 12.2 Drop 'ETH7' due to its irrevelancy
df = df.drop(columns=['ETH7'])


# 13. DATA PARTITION AND STANDARDIZATION
# 13.1 Data Partition into subsets
Value = df[['INCOME',  'MINRAMNT', 'AVGGIFT', 'CARDPROM','NUMPROM','MDMAUD_A_C','HIT','TIMELAG',
'NUMPRM12', 'LASTGIFT', 'MAJOR', 'LASTDATE', 'MAXADATE', 'WEALTH2',
'MDMAUD_R_C', 'MDMAUD_R_L', 'MDMAUD_R_D', 'MDMAUD_R_I', 'MDMAUD_F_1',
'MDMAUD_F_2', 'MDMAUD_F_5', 'MDMAUD_A_L', 'MDMAUD_A_M',
'MDMAUD_A_T']]

Social = df[['MALEVET', 'LOCALGOV', 'STATEGOV', 'FEDGOV','AFC4', 'AFC5',
'AFC6', 'VC1', 'VC2', 'VC3', 'VC4', 'AGE', 'HOMEOWNR', 'PEPSTRFL', 'MALE', 'OTHER',
'DOMAIN_U', 'DOMAIN_S', 'DOMAIN_T', 'DOMAIN_R', 'DOMAIN_C']]

# 13.1.1 Correlations into Value
# Verifing the correlations into 'Value' to help in the selection of the variables
plt.rcParams['figure.figsize'] = (19,19)
corr_matrix_1=Value.corr(method = 'spearman')
mask=np.zeros_like(corr_matrix_1, dtype=np.bool)
mask[np.triu_indices_from(mask)]=True
sns.heatmap(data=corr_matrix_1, mask=mask, center=0, annot=True, linewidths=2, cmap='coolwarm')
plt.tight_layout()

# Verifing the correlations into 'Social' to help in the selection of the variables
plt.rcParams['figure.figsize'] = (19,19)
corr_matrix_2=Social.corr(method = 'spearman')
mask=np.zeros_like(corr_matrix_2, dtype=np.bool)
mask[np.triu_indices_from(mask)]=True
sns.heatmap(data=corr_matrix_2, mask=mask, center=0, annot=True, linewidths=2, cmap='coolwarm')
plt.tight_layout()

# 13.2. Data Standardization 
# 13.2.1 Select from Value the numeric data
Value_df = Value[['INCOME',  'MINRAMNT', 'AVGGIFT', 'CARDPROM',
'NUMPRM12', 'LASTGIFT', 'MAJOR']]

# 13.2.2 Standardize Value
scaler = StandardScaler()
std_value = scaler.fit_transform(Value_df)
std_value = pd.DataFrame(std_value, columns = Value_df.columns).set_index(Value_df.index)

# 13.2.3 Standardize Social
Social_df = Social[['MALEVET', 'LOCALGOV', 'STATEGOV', 'FEDGOV','AFC4', 'AFC5',
'AFC6', 'VC1', 'VC2', 'VC3', 'VC4', 'AGE']]
scaler = StandardScaler()
std_social = scaler.fit_transform(Social_df)
std_social = pd.DataFrame(std_social, columns = Social_df.columns).set_index(Social_df.index)

# 13.3 Join the categorical features into the dataframes 
Value_data = pd.concat([std_value, df[['MAJOR','WEALTH2','MDMAUD_R_C', 'MDMAUD_R_L', 'MDMAUD_R_D', 'MDMAUD_R_I', 'MDMAUD_F_1',
'MDMAUD_F_2', 'MDMAUD_F_5', 'MDMAUD_A_L', 'MDMAUD_A_M',
'MDMAUD_A_T'
]]], axis=1, sort=False, join='inner')

# 13.3.1 Create the categorical datafame for Value
Value_cat = df[['MAJOR','WEALTH2','MDMAUD_R_C', 'MDMAUD_R_L', 'MDMAUD_R_D', 'MDMAUD_R_I', 'MDMAUD_F_1',
'MDMAUD_F_2', 'MDMAUD_F_5', 'MDMAUD_A_L', 'MDMAUD_A_M',
'MDMAUD_A_T']]

# 13.4 Join the categorical features into the dataframes 
Social_data = pd.concat([std_social, df[['HOMEOWNR', 'PEPSTRFL', 'MALE', 'OTHER',
'DOMAIN_U', 'DOMAIN_S', 'DOMAIN_T', 'DOMAIN_R','DOMAIN_C']]], axis=1, sort=False, join='inner')

# 13.4.1 Create the categorical datafame for Social
Social_cat = df[['HOMEOWNR', 'PEPSTRFL', 'MALE', 'OTHER',
'DOMAIN_U', 'DOMAIN_S', 'DOMAIN_T', 'DOMAIN_R','DOMAIN_C']]

# 14. VALUE ANALYSIS
# K-MEANS CLUSTERING FOLLOWED BY HIERARQUICAL CLUSTERING

std_value=std_value[['INCOME', 'MINRAMNT',  'LASTGIFT', 'CARDPROM','NUMPRM12']] 

k = 100
# 14.1 Apply algorithm with k clusters
kmeans_value = KMeans(n_clusters = k, init = 'k-means++', n_init = 65, max_iter = 300).fit(std_value)

# 14.1.1 Cluster Evaluation
# Get centroids and save them as a dataframe 
centroids_value = kmeans_value.cluster_centers_
centroids_value = pd.DataFrame(centroids_value, columns = std_value.columns)
clusters_value = pd.DataFrame(kmeans_value.labels_, columns=['Centroids'])
clusters_value['ID'] = Value_df.index

# 14.1.2 Hierarchical clustering on top of K-means
# Plot the dendrogram to decide number of clusters
pyplot.figure(figsize=(10, 7))  
pyplot.title("Dendrogram for Value")  
dend = shc.dendrogram(shc.linkage(centroids_value,method='ward'))

# 14.2 Apply agglomerative clustering
# Apply hierarchical clustering with number of clusters defined and best method (ward)
Hclustering = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
my_HC = Hclustering.fit(centroids_value)
my_labels = pd.DataFrame(my_HC.labels_).reset_index()
my_labels.columns =  ['Centroids','Cluster']
# Check number of k-means centroids by hierachical clusters
count_centroids= my_labels.groupby(by='Cluster')['Cluster'].count().reset_index(name='N')

# Create 'final_df' with column 'Cluster' identifying to each cluster the observations belong to
final_df = clusters_value.merge(my_labels, how='inner', on='Centroids')
final_df = std_value.merge(final_df, how='inner', left_on=std_value.index, right_on='ID')
final_df = final_df.set_index('ID')
final_df = final_df.drop(columns='Centroids')
# Centroids from the hierarchical clustering 
centroids_HC = final_df.groupby(['Cluster']).mean()
# Check number of observations by cluster
count_HC = final_df.Cluster.value_counts()
count_HC = final_df.groupby(by='Cluster')['Cluster'].count().reset_index(name='N')

count_clusters_value = count_HC.copy()
print('VALUES FOR SEGMENT VALUE', count_clusters_value)


# 15. SOCIAL ANALYSIS
# K-MEANS CLUSTERING FOLLOWED BY HIERARQUICAL CLUSTERING
new_std_social1=std_social[['AFC4','AFC5','FEDGOV','VC1']]
k = 100

# 15.1 Apply algorithm with k clusters
kmeans_social = KMeans(n_clusters = k, init = 'k-means++', n_init = 65, max_iter = 300).fit(new_std_social1)

# 15.1.1 Cluster Evaluation
# Get centroids and save them as a dataframe 
centroids_social = kmeans_social.cluster_centers_
centroids_social = pd.DataFrame(centroids_social, columns = new_std_social1.columns)
clusters_social = pd.DataFrame(kmeans_social.labels_, columns=['Centroids'])
clusters_social['ID'] = Social_df.index


# 15.1.2 Hierarchical clustering on top of K-means
# Plot the dendrogram to decide number of clusters
pyplot.figure(figsize=(10, 7))  
pyplot.title("Dendrogram for Social")  
dend = shc.dendrogram(shc.linkage(centroids_social,method='ward'))

# 15.2 Apply agglomerative clustering
# Aplly hierarchical clustering with number of clusters defined and best method (ward)
Hclustering = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
my_HC = Hclustering.fit(centroids_social)
my_labels = pd.DataFrame(my_HC.labels_).reset_index()
my_labels.columns =  ['Centroids','Cluster']
# Check number of k-means centroids by hierachical clusters
count_centroids= my_labels.groupby(by='Cluster')['Cluster'].count().reset_index(name='N')

# Create 'final_df' with column 'Cluster' identifying to each cluster the observations belong to
final_df_social = clusters_social.merge(my_labels, how='inner', on='Centroids')
final_df_social = std_social.merge(final_df_social, how='inner', left_on=std_social.index, right_on='ID')
final_df_social = final_df_social.set_index('ID')
final_df_social = final_df_social.drop(columns='Centroids')
# Centroids from the hierarchical clustering 
centroids_HC = final_df_social.groupby(['Cluster']).mean()
# Check number of observations by cluster
count_HC = final_df_social.Cluster.value_counts()
count_HC = final_df_social.groupby(by='Cluster')['Cluster'].count().reset_index(name='N')

count_clusters_social = count_HC.copy()
print('VALUES FOR SEGMENT SOCIAL', count_clusters_social)



# 16. K-MODE for Social
# 16.1. Elbow Graph
# cost will append the error/inertia of the application of K-modes to the 'value modes' dataset with 1 to max_k number of clusters
cost = [] 
max_k = 10
# 'Value Modes' is a dataset with the non-numerical variables to use in K-modes
social_modes = Social_cat.astype('str')
cost = []
for num_clusters in list(range(1,5)):
    kmode = KModes(n_clusters=num_clusters, init = "Cao", n_init = 1, verbose=1)
    kmode.fit_predict(social_modes)
    cost.append(kmode.cost_)

y = np.array([i for i in range(1,5,1)])
plt.plot(y,cost)

# Choosing K=3
km_cao = KModes(n_clusters=3, init = "Cao", n_init = 75, verbose=1)
clusters_kmodes = km_cao.fit_predict(social_modes)
# Combining the predicted clusters with the original DF.
value_kmodes = social_modes.copy()

clustersDf = pd.DataFrame(clusters_kmodes)
clustersDf.columns = ['cluster_predicted_kmodes']
combinedDf = pd.concat([value_kmodes, clustersDf], axis = 1).reset_index()
combinedDf = combinedDf.drop(['index'], axis = 1)
# Cluster identification
cluster_0 = combinedDf[combinedDf['cluster_predicted_kmodes'] == 0]
cluster_1 = combinedDf[combinedDf['cluster_predicted_kmodes'] == 1]
cluster_2 = combinedDf[combinedDf['cluster_predicted_kmodes'] == 2]

clusters_kmodes = pd.DataFrame(clusters_kmodes)
clusters_kmodes.columns = ['Cluster_Kmodes']
clusters_kmodes['ID'] = social_modes.index
clusters_kmodes.set_index('ID',inplace = True)


f, axs = plt.subplots(2,4,figsize = (15,5)) 
plt.subplots_adjust(wspace = 0.3, hspace = 0.3)  
sns.countplot(x=combinedDf['HOMEOWNR'],order=combinedDf['HOMEOWNR'].value_counts().index,hue=combinedDf['cluster_predicted_kmodes'],ax=axs[0,0])
sns.countplot(x=combinedDf['PEPSTRFL'],order=combinedDf['PEPSTRFL'].value_counts().index,hue=combinedDf['cluster_predicted_kmodes'],ax=axs[0,1])
sns.countplot(x=combinedDf['MALE'],order=combinedDf['MALE'].value_counts().index,hue=combinedDf['cluster_predicted_kmodes'],ax=axs[0,2])
sns.countplot(x=combinedDf['OTHER'],order=combinedDf['OTHER'].value_counts().index,hue=combinedDf['cluster_predicted_kmodes'],ax=axs[0,3])
sns.countplot(x=combinedDf['DOMAIN_U'],order=combinedDf['DOMAIN_U'].value_counts().index,hue=combinedDf['cluster_predicted_kmodes'],ax=axs[1,0])
sns.countplot(x=combinedDf['DOMAIN_S'],order=combinedDf['DOMAIN_S'].value_counts().index,hue=combinedDf['cluster_predicted_kmodes'],ax=axs[1,1])
sns.countplot(x=combinedDf['DOMAIN_T'],order=combinedDf['DOMAIN_T'].value_counts().index,hue=combinedDf['cluster_predicted_kmodes'],ax=axs[1,2])
sns.countplot(x=combinedDf['DOMAIN_R'],order=combinedDf['DOMAIN_R'].value_counts().index,hue=combinedDf['cluster_predicted_kmodes'],ax=axs[1,3])
plt.tight_layout()
plt.title('K-Modes for Social')
plt.show()



# 17. DATA VISUALIZATION
# 17.1 Radar Chart to interpert Cluster centoids for Value
graph1 = final_df.groupby('Cluster')['INCOME','MINRAMNT','LASTGIFT','CARDPROM','NUMPRM12'].mean().reset_index()
graph1 = graph1.melt(id_vars='Cluster', var_name='Variable', value_name='Average')
graph1['Cluster'] = graph1['Cluster'].astype('int64')
fig = pe.line_polar(graph1, r="Average", theta="Variable", color="Cluster", line_close=True, color_discrete_sequence=["blue", "green", "red"])
plot(fig)

# 17.2 Radar Chart to interpert Cluster centoids for Social
graph2 = final_df_social.groupby('Cluster')['AFC4','AFC5','FEDGOV','VC1'].mean().reset_index()
graph2 = graph2.melt(id_vars='Cluster', var_name='Variable', value_name='Average')
graph2['Cluster'] = graph2['Cluster'].astype('int64')
fig = pe.line_polar(graph2, r="Average", theta="Variable", color="Cluster", line_close=True, color_discrete_sequence=["blue", "green", "red"])
plot(fig)

