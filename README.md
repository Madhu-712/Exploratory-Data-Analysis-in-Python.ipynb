# Exploratory-Data-Analysis-in-Python.ipynb

Exploratory Data Analysis (EDA) with Python
This is an exploratory data analysis project. In this project, I explore the Absenteeism time in hours dataset.

It is categorized into various sections which are listed in table of contents as follows:-

Table of contents:-
Introduction to EDA
Distribution of a variable
Types of EDA
Objectives of EDA
Exploratory data analysis – prerequisites
Import the required Python libraries
The dataset description
Import the dataset
Overview of the dataset
Check for anomalies in the dataset
Univariate analysis
Multivariate analysis
1. Introduction to EDA
Several questions come to mind when we come across a new dataset. The below list shed light on some of these questions:-

• What is the distribution of the dataset?

• Are there any missing numerical values, outliers or anomalies in the dataset?

• What are the underlying assumptions in the dataset?

• Whether there exists relationships between variables in the dataset?

• How to be sure that our dataset is ready for input in a machine learning algorithm?

• How to select the most suitable algorithm for a given dataset?

So, how do we get answer to the above questions?

The answer is Exploratory Data Analysis. It enable us to answer all of the above questions.

Exploratory Data Analysis or EDA is a critical first step in analyzing a new dataset. The primary objective of EDA is to analyze the data for distribution, outliers and anomalies in the dataset. It enable us to direct specific testing of the hypothesis. It includes analysing the data to find the distribution of data, its main characteristics, identifying patterns and visualizations. It also provides tools for hypothesis generation by visualizing and understanding the data through graphical representation.

2. Distribution of a variable
There are three types of distribution of a variable. They are Univariate, Bivariate and Multivariate distribution. Variables mean the number of objects that are under consideration as a sample in an experiment.

Univariate distribution

In univariate distribution, there is only one variable under consideration. It is the simplest form of analysis because only one quantity changes. It does not deal with causes or relationships. The main purpose of the analysis is to describe the data and find patterns that exist within it. We can describe patterns found in univariate data using central tendency (mean, median and mode) and dispersion (range, variance, standard deviation, maximum and minimum values and interquartile range). We can visualize the univariate data using various types of charts and graphs. These are frequency distribution tables, histograms, bar charts, pie charts and frequency polygons.

Bivariate distribution

This type of data distribution involves two different variables. The analysis of this type of data deals with causes and relationships and the analysis is done to find out the relationship among the two variables. A very common example of bivariate distribution is height and weight of a single person.

Bivariate analysis means the analysis of bivariate data. It is one of the simplest forms of statistical analysis, used to find out if there is a relationship between two sets of values. Thus bivariate data analysis involves comparisons, exploring relationships, finding causes and explanations. These variables are often plotted on X and Y axis on the graph for better understanding of data and one of these variables is independent while the other is dependent.

Common types of bivariate analysis include drawing scatter plot, regression analysis and finding correlation coefficients. A scatter plot is used to find out if there exists any relationship between two variables. Regression analysis is a statistical method for estimating the relationships between variables. Correlation coefficient analysis measures the strength and direction of a linear relationship between two variables on a scatter plot.

Multivariate distribution

When the dataset involves three or more variables, it is categorized under multivariate distribution. Multivariate analysis is used to study more complex sets of data. It is usually unsuitable for small sets of data.

There are wide variety of analysis techniques to perform multivariate analysis. The choice of analysis techniques depends on the dataset and our goals to be achieved. Some examples of multivariate analysis techniques are additive tree, cluster analysis, correspondence analysis, factor analysis, MANOVA (multivariate analysis of variance), multidimensional scaling, multiple regression analysis, principal component analysis and redundancy analysis.

3. Types of EDA
EDA is generally cross-classified in two ways. First, each method is either non-graphical or graphical. Second, each method is either univariate or multivariate (usually bivariate). The non-graphical methods provide insight into the characteristics and the distribution of the variable(s) of interest. So, non-graphical methods involve calculation of summary statistics while graphical methods include summarizing the data diagrammatically.

There are four types of exploratory data analysis (EDA) based on the above cross-classification methods. Each of these types of EDA are described below:-

i. Univariate non-graphical EDA
The objective of the univariate non-graphical EDA is to understand the sample distribution and also to make some initial conclusions about population distributions. Outlier detection is also a part of this analysis.

ii. Multivariate non-graphical EDA
Multivariate non-graphical EDA techniques show the relationship between two or more variables in the form of either cross-tabulation or statistics.

iii. Univariate graphical EDA
In addition to finding the various sample statistics of univariate distribution (discussed above), we also look graphically at the distribution of the sample. The non-graphical methods are quantitative and objective. They do not give full picture of the data. Hence, we need graphical methods, which are more qualitative in nature and presents an overview of the data.

iv. Multivariate graphical EDA
There are several useful multivariate graphical EDA techniques, which are used to look at the distribution of multivariate data. These are as follows:-

Side-by-Side Boxplots

Scatterplots

Curve Fitting

Heat Maps and 3-D Surface Plots

4. Objectives of EDA
The objectives of the EDA are as follows:-

i. To get an overview of the distribution of the dataset.

ii. Check for missing numerical values, outliers or other anomalies in the dataset.

iii.Discover patterns and relationships between variables in the dataset.

iv. Check the underlying assumptions in the dataset.

5. Exploratory data analysis - prerequisites
We need two Python libraries for exploratory data analysis – NumPy and Pandas.

• NumPy – NumPy is the fundamental Python library for scientific computing. It adds support for large and multi-dimensional arrays and matrices. It also supports large collection of high-level mathematical functions to operate on these arrays.

• Pandas - Pandas is a software library for Python programming language which provide tools for data manipulation and analysis tasks. It will enable us to manipulate numerical tables and time series using data structures and operations.

We need two more libraries for data visualization purpose. These are Seaborn and Matplotlib.

• Seaborn - Seaborn is a Python data visualization library based on Matplotlib. It provides a high level interface for drawing attractive and informative statistical graphics.

• Matplotlib - Matplotlib is the core data visualization library of Python programming language. It provides an object-oriented API for embedding plots into applications.

6. Import the required Python libraries
We have seen that we need two Python libraries – NumPy and Pandas for the exploratory data analysis process. Also, we need two more libraries – Seaborn and Matplotlib for data visualization purposes.

We need to import these libraries before we actually start using them. We can import them with their usual shorthand notation as follows:-

# ignore the warnings

import warnings
warnings.simplefilter(action = "ignore", category = FutureWarning)
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as st
%matplotlib inline
sns.set(style="whitegrid")
7. The dataset description
For this project, I have used the Absenteeism at work dataset. This dataset can be found at the following url-

https://archive.ics.uci.edu/ml/datasets/Absenteeism+at+work

The dataset consists of records of absenteeism at work from July 2007 to July 2010 at a courier company in Brazil. The dataset contains 740 number of instances and 21 number of attributes. It was created by Andrea Martiniano, Ricardo Pinto Ferreira and Renato Jose Sassi.

Attribute information in the dataset is as follows:-

ID – represents individual identification ID
Reason for absence (ICD) – Absences attested by the International Code of Diseases (ICD) stratified into 21 categories (I to XXI) as follows:
I Certain infectious and parasitic diseases

II Neoplasms

III Diseases of the blood and blood-forming organs and certain disorders involving the immune mechanism

IV Endocrine, nutritional and metabolic diseases

V Mental and behavioural disorders

VI Diseases of the nervous system

VII Diseases of the eye and adnexa

VIII Diseases of the ear and mastoid process

IX Diseases of the circulatory system

X Diseases of the respiratory system

XI Diseases of the digestive system

XII Diseases of the skin and subcutaneous tissue

XIII Diseases of the musculoskeletal system and connective tissue

XIV Diseases of the genitourinary system

XV Pregnancy, childbirth and the puerperium

XVI Certain conditions originating in the perinatal period

XVII Congenital malformations, deformations and chromosomal abnormalities

XVIII Symptoms, signs and abnormal clinical and laboratory findings, not elsewhere classified

XIX Injury, poisoning and certain other consequences of external causes

XX External causes of morbidity and mortality

XXI Factors influencing health status and contact with health services.

And 7 categories without (CID) patient follow-up (22), medical consultation (23), blood donation (24), laboratory examination (25), unjustified absence (26), physiotherapy (27), dental consultation (28).

Month of absence
Day of the week (Monday (2), Tuesday (3), Wednesday (4), Thursday (5), Friday (6))
Seasons (summer (1), autumn (2), winter (3), spring (4))
Transportation expense
Distance from Residence to Work (kilometers)
Service time
Age
Work load Average/day
Hit target
Disciplinary failure (yes=1; no=0)
Education (high school (1), graduate (2), postgraduate (3), master and doctor (4))
Son (number of children)
Social drinker (yes=1; no=0)
Social smoker (yes=1; no=0)
Pet (number of pet)
Weight
Height
Body mass index
Absenteeism time in hours (target)
8. Import the dataset
We can import the dataset using the usual read_csv() function as follows:-

data = "C:/eda/Absenteeism_at_work.csv"

df = pd.read_csv(data, sep=";")
Generally, in the csv file the values are separated by a comma. So, there is no need to use the sep parameter which describes how the values are separated. In this case, the values are separated by semicolon (;). So, I used the sep = ";" parameter to denote that the values are separated by semicolon.

9. Overview of the dataset
Now, we should get to know our data. We should know its dimensions, structure and column data types.

We can proceed as follows:-

df.shape attribute
The first thing that I do is to check the dimensions of the dataset. We can check the dimensions of the data with df.shape attribute as follows:-

print(df.shape)
(740, 21)
Interpretation

We can see that our dataset has 740 rows and 21 columns.

df.columns attribute
I can view the column names in the dataset with df.columns attribute as follows:-

print(df.columns)
Index(['ID', 'Reason for absence', 'Month of absence', 'Day of the week',
       'Seasons', 'Transportation expense', 'Distance from Residence to Work',
       'Service time', 'Age', 'Work load Average/day ', 'Hit target',
       'Disciplinary failure', 'Education', 'Son', 'Social drinker',
       'Social smoker', 'Pet', 'Weight', 'Height', 'Body mass index',
       'Absenteeism time in hours'],
      dtype='object')
df.head() and df.tail() methods
Now, it is time to get an overview of the dataset. We can view the top five and bottom five rows of the dataset with df.head() and df.tail() methods respectively.

So, we proceed as follows:-

df.head()
ID	Reason for absence	Month of absence	Day of the week	Seasons	Transportation expense	Distance from Residence to Work	Service time	Age	Work load Average/day	...	Disciplinary failure	Education	Son	Social drinker	Social smoker	Pet	Weight	Height	Body mass index	Absenteeism time in hours
0	11	26	7	3	1	289	36	13	33	239.554	...	0	1	2	1	0	1	90	172	30	4
1	36	0	7	3	1	118	13	18	50	239.554	...	1	1	1	1	0	0	98	178	31	0
2	3	23	7	4	1	179	51	18	38	239.554	...	0	1	0	1	0	0	89	170	31	2
3	7	7	7	5	1	279	5	14	39	239.554	...	0	1	2	1	1	0	68	168	24	4
4	11	23	7	5	1	289	36	13	33	239.554	...	0	1	2	1	0	1	90	172	30	2
5 rows × 21 columns

df.tail()
ID	Reason for absence	Month of absence	Day of the week	Seasons	Transportation expense	Distance from Residence to Work	Service time	Age	Work load Average/day	...	Disciplinary failure	Education	Son	Social drinker	Social smoker	Pet	Weight	Height	Body mass index	Absenteeism time in hours
735	11	14	7	3	1	289	36	13	33	264.604	...	0	1	2	1	0	1	90	172	30	8
736	1	11	7	3	1	235	11	14	37	264.604	...	0	3	1	0	0	1	88	172	29	4
737	4	0	0	3	1	118	14	13	40	271.219	...	0	1	1	1	0	8	98	170	34	0
738	8	0	0	4	2	231	35	14	39	271.219	...	0	1	2	1	0	2	100	170	35	0
739	35	0	0	6	3	179	45	14	53	271.219	...	0	1	1	0	0	1	77	175	25	0
5 rows × 21 columns

df.info() method
We can get a concise summary of the dataset with df.info() method. This method prints information about a dataFrame including the index, column names and data types, non-null values and memory usage.

df.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 740 entries, 0 to 739
Data columns (total 21 columns):
ID                                 740 non-null int64
Reason for absence                 740 non-null int64
Month of absence                   740 non-null int64
Day of the week                    740 non-null int64
Seasons                            740 non-null int64
Transportation expense             740 non-null int64
Distance from Residence to Work    740 non-null int64
Service time                       740 non-null int64
Age                                740 non-null int64
Work load Average/day              740 non-null float64
Hit target                         740 non-null int64
Disciplinary failure               740 non-null int64
Education                          740 non-null int64
Son                                740 non-null int64
Social drinker                     740 non-null int64
Social smoker                      740 non-null int64
Pet                                740 non-null int64
Weight                             740 non-null int64
Height                             740 non-null int64
Body mass index                    740 non-null int64
Absenteeism time in hours          740 non-null int64
dtypes: float64(1), int64(20)
memory usage: 121.5 KB
Interpretation

We can see that this method prints information about all columns. There are no missing values in the dataset. We need to confirm this further.

The data types of several columns like "Month of absence", "Transportation expense", "Distance from Residence to work", "Hit target", "Education", "Weight", "Height", "Body mass index" and "Absenteeism time in hours" should be real or float. But, the above cell shows that they have integer data types. So, we need to convert their data types into float.

We can do it as follows:-

df[["Month of absence","Transportation expense","Distance from Residence to Work", "Hit target","Education","Weight","Height",
"Body mass index","Absenteeism time in hours"]]=df[["Month of absence","Transportation expense","Distance from Residence to Work",
"Hit target","Education","Weight","Height","Body mass index","Absenteeism time in hours"]].astype(float)
We should again check the data types of modified columns.

df.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 740 entries, 0 to 739
Data columns (total 21 columns):
ID                                 740 non-null int64
Reason for absence                 740 non-null int64
Month of absence                   740 non-null float64
Day of the week                    740 non-null int64
Seasons                            740 non-null int64
Transportation expense             740 non-null float64
Distance from Residence to Work    740 non-null float64
Service time                       740 non-null int64
Age                                740 non-null int64
Work load Average/day              740 non-null float64
Hit target                         740 non-null float64
Disciplinary failure               740 non-null int64
Education                          740 non-null float64
Son                                740 non-null int64
Social drinker                     740 non-null int64
Social smoker                      740 non-null int64
Pet                                740 non-null int64
Weight                             740 non-null float64
Height                             740 non-null float64
Body mass index                    740 non-null float64
Absenteeism time in hours          740 non-null float64
dtypes: float64(10), int64(11)
memory usage: 121.5 KB
Interpretation

We can see that the data types of modified columns are float64. Now, all the columns are appropriate data types.

drop redundant columns
There are two columns ID and Pet which have no correlation with the target variable Absenteeism time in hours. So, we should drop these columns. We can do it as follows:-

df.drop(['ID','Pet'], axis = 1, inplace=True)
df.describe() method
We can view the summary statistics of numerical columns with df.describe() method. It enable us to detect outliers in the data which require further investigation.

print(df.describe())
       Reason for absence  Month of absence  Day of the week     Seasons  \
count          740.000000        740.000000       740.000000  740.000000   
mean            19.216216          6.324324         3.914865    2.544595   
std              8.433406          3.436287         1.421675    1.111831   
min              0.000000          0.000000         2.000000    1.000000   
25%             13.000000          3.000000         3.000000    2.000000   
50%             23.000000          6.000000         4.000000    3.000000   
75%             26.000000          9.000000         5.000000    4.000000   
max             28.000000         12.000000         6.000000    4.000000   

       Transportation expense  Distance from Residence to Work  Service time  \
count              740.000000                       740.000000    740.000000   
mean               221.329730                        29.631081     12.554054   
std                 66.952223                        14.836788      4.384873   
min                118.000000                         5.000000      1.000000   
25%                179.000000                        16.000000      9.000000   
50%                225.000000                        26.000000     13.000000   
75%                260.000000                        50.000000     16.000000   
max                388.000000                        52.000000     29.000000   

              Age  Work load Average/day   Hit target  Disciplinary failure  \
count  740.000000              740.000000  740.000000            740.000000   
mean    36.450000              271.490235   94.587838              0.054054   
std      6.478772               39.058116    3.779313              0.226277   
min     27.000000              205.917000   81.000000              0.000000   
25%     31.000000              244.387000   93.000000              0.000000   
50%     37.000000              264.249000   95.000000              0.000000   
75%     40.000000              294.217000   97.000000              0.000000   
max     58.000000              378.884000  100.000000              1.000000   

        Education         Son  Social drinker  Social smoker      Weight  \
count  740.000000  740.000000      740.000000     740.000000  740.000000   
mean     1.291892    1.018919        0.567568       0.072973   79.035135   
std      0.673238    1.098489        0.495749       0.260268   12.883211   
min      1.000000    0.000000        0.000000       0.000000   56.000000   
25%      1.000000    0.000000        0.000000       0.000000   69.000000   
50%      1.000000    1.000000        1.000000       0.000000   83.000000   
75%      1.000000    2.000000        1.000000       0.000000   89.000000   
max      4.000000    4.000000        1.000000       1.000000  108.000000   

           Height  Body mass index  Absenteeism time in hours  
count  740.000000       740.000000                 740.000000  
mean   172.114865        26.677027                   6.924324  
std      6.034995         4.285452                  13.330998  
min    163.000000        19.000000                   0.000000  
25%    169.000000        24.000000                   2.000000  
50%    170.000000        25.000000                   3.000000  
75%    172.000000        31.000000                   8.000000  
max    196.000000        38.000000                 120.000000  
Interpretation

We can see that the minimum value of Month of absence is zero. It cannot be zero. The minimum value should be one. So, we need to replace zero by one. We can do it as follows:-

df['Month of absence'].replace(0,1,inplace=True)
print(df.describe())
       Reason for absence  Month of absence  Day of the week     Seasons  \
count          740.000000        740.000000       740.000000  740.000000   
mean            19.216216          6.328378         3.914865    2.544595   
std              8.433406          3.429397         1.421675    1.111831   
min              0.000000          1.000000         2.000000    1.000000   
25%             13.000000          3.000000         3.000000    2.000000   
50%             23.000000          6.000000         4.000000    3.000000   
75%             26.000000          9.000000         5.000000    4.000000   
max             28.000000         12.000000         6.000000    4.000000   

       Transportation expense  Distance from Residence to Work  Service time  \
count              740.000000                       740.000000    740.000000   
mean               221.329730                        29.631081     12.554054   
std                 66.952223                        14.836788      4.384873   
min                118.000000                         5.000000      1.000000   
25%                179.000000                        16.000000      9.000000   
50%                225.000000                        26.000000     13.000000   
75%                260.000000                        50.000000     16.000000   
max                388.000000                        52.000000     29.000000   

              Age  Work load Average/day   Hit target  Disciplinary failure  \
count  740.000000              740.000000  740.000000            740.000000   
mean    36.450000              271.490235   94.587838              0.054054   
std      6.478772               39.058116    3.779313              0.226277   
min     27.000000              205.917000   81.000000              0.000000   
25%     31.000000              244.387000   93.000000              0.000000   
50%     37.000000              264.249000   95.000000              0.000000   
75%     40.000000              294.217000   97.000000              0.000000   
max     58.000000              378.884000  100.000000              1.000000   

        Education         Son  Social drinker  Social smoker      Weight  \
count  740.000000  740.000000      740.000000     740.000000  740.000000   
mean     1.291892    1.018919        0.567568       0.072973   79.035135   
std      0.673238    1.098489        0.495749       0.260268   12.883211   
min      1.000000    0.000000        0.000000       0.000000   56.000000   
25%      1.000000    0.000000        0.000000       0.000000   69.000000   
50%      1.000000    1.000000        1.000000       0.000000   83.000000   
75%      1.000000    2.000000        1.000000       0.000000   89.000000   
max      4.000000    4.000000        1.000000       1.000000  108.000000   

           Height  Body mass index  Absenteeism time in hours  
count  740.000000       740.000000                 740.000000  
mean   172.114865        26.677027                   6.924324  
std      6.034995         4.285452                  13.330998  
min    163.000000        19.000000                   0.000000  
25%    169.000000        24.000000                   2.000000  
50%    170.000000        25.000000                   3.000000  
75%    172.000000        31.000000                   8.000000  
max    196.000000        38.000000                 120.000000  
Now, the Month of absence column has minimum value one.

10. Check for anomalies in the dataset
Now, we should check for any discrepancy in the dataset.

Check for missing numerical values
The first step is to check for any missing values in the dataset. We can check for missing values in the dataset using the df.isnull().sum() command. This command returns the total number of missing values in each column in the dataset.

If we want to check for 'NA' values in a particular column in the dataframe, then we should use the following command pd.isna(df['col_name']).

We can proceed as follows:-

df.isnull().sum()
Reason for absence                 0
Month of absence                   0
Day of the week                    0
Seasons                            0
Transportation expense             0
Distance from Residence to Work    0
Service time                       0
Age                                0
Work load Average/day              0
Hit target                         0
Disciplinary failure               0
Education                          0
Son                                0
Social drinker                     0
Social smoker                      0
Weight                             0
Height                             0
Body mass index                    0
Absenteeism time in hours          0
dtype: int64
Interpretation

The above command shows that there are no missing values in the dataset.

Check with ASSERT statement
We should confirm that our dataset has no missing values. We can write an assert statement to verify this. We can use an assert statement to programmatically check that no missing, unexpected 0 or negative values are present. This gives us confidence that our code is running properly.

Assert statement will return nothing if the value being tested is true and will throw an AssertionError if the value is false.

Asserts

• assert 1 == 1 (return Nothing if the value is True)

• assert 1 == 2 (return AssertionError if the value is False)

#assert that there are no missing values in the dataframe

assert pd.notnull(df).all().all()
#assert all values are greater than or equal to 0

assert (df >= 0).all().all()
Interpretation

The above two commands do not throw any error. Hence, it is confirmed that there are no missing or negative values in the dataset. All the values are greater than or equal to zero.

11. Univariate analysis
Measures of central tendency and dispersion
Central tendency means a central value which describe a probability distribution. It may also be called a center or location of the distribution. The most common measures of central tendency are the arithmetic mean, the median and the mode. The most common measure of central tendency is the mean. For skewed distribution or when there is concern about outliers, the median may be preferred. So, median is more robust measure than the mean.

Dispersion is an indicator of how far away from the center, we can find the data values. The most common measures of dispersion are variance, standard deviation and interquartile range(IQR). Variance is the standard measure of spread. The standard deviation is the square root of the variance. The variance and standard deviation are two useful measures of spread.

A third measure of spread is the interquartile range (IQR). The IQR is calculated using the boundaries of data situated between the 1st and the 3rd quartiles. So, IQR can be calculated as IQR = Q3 - Q1. It is a robust measure of spread.

The above measures can be calculated by df.describe() method as follows:-

print(df['Absenteeism time in hours'].describe())
count    740.000000
mean       6.924324
std       13.330998
min        0.000000
25%        2.000000
50%        3.000000
75%        8.000000
max      120.000000
Name: Absenteeism time in hours, dtype: float64
Interpretation

The count, min and max values represent the number of counts, minimum and maximum values of the target variable Absenteeism time in hours.

The measures of central tendency are given by the mean(6.924324) and median(50% value-3.00).

The measure of dispersion is given by the standard deviation given by std(13.330998).

The 25%, 50% and 75% values show the corresponding percentiles. 50th percentile denote the median of the distribution.

The IQR is the difference between 75th and 25th percentiles. Hence, IQR = 8.00 - 2.00 = 6.00

Measures of shape
We have looked at the measures of central tendency of the data (mean and median) and spread of the data (standard deviation(std), interquartile range, minimum (min) and maximum (max) values. These quantities can only be used for quantitative variables not for categorical variables.

Now, we will take a look at measures of shape of distribution. There are two statistical measures that can tell us about the shape of the distribution. These measures are skewness and kurtosis. These measures can be used to convey information about the shape of the distribution of the dataset.

First, we will look at skewness and later we get to know about kurtosis.

Skewness
Skewness is a measure of a distribution's symmetry or more precisely lack of symmetry. It is used to mean the absence of symmetry from the mean of the dataset. It is a characteristic of the deviation from the mean. It is used to indicate the shape of the distribution of data.

Negative skewness
Negative values for skewness indicate negative skewness. In this case, the data are skewed or tail to left. By skewed left, we mean that the left tail is long relative to the right tail. The data values may extend further to the left but concentrated in the right. So, there is a long tail and distortion is caused by extremely small values which pull the mean downward so that it is less than the median. Hence, in this case

Mean < Median < Mode

Zero skewness
Zero skewness means skewness value of zero. It means the dataset is symmetrical. A data set is symmetrical if it looks the same to the left and right to the center point. The dataset looks bell shaped or symmetrical. A perfectly symmetrical data set will have a skewness of zero. So, the normal distribution which is perfectly symmetrical has a skewness of 0. So, in this case

Mean = Median = Mode

Positive skewness
Positive values for skewness indicate positive skewness. The dataset are skewed or tail to right. By skewed right, we mean that the right tail is long relative to the left tail. The data values are concentrated in the right. So, there is a long tail to the right that is caused by extremely large values which pull the mean upward so that it is greater than the median. So, we have

Mean > Median > Mode

Reference range on skewness values
The rule of thumb for skewness values are:

If the skewness is between -0.5 and 0.5, the data are fairly symmetrical.

If the skewness is between -1 and – 0.5 or between 0.5 and 1, the data are moderately skewed.

If the skewness is less than -1 or greater than 1, the data are highly skewed.

We can proceed as follows:-

df['Absenteeism time in hours'].skew()
5.720727863123873
Interpretation of skewness

The skewness of our target variable Absenteeism time in hours comes out to be greater than +1. So, we can conclude that the target variable is highly positively skewed.

We can confirm this by plotting a Seaborn distplot diagram as follows:-

plt.figure(figsize= (10,8))
sns.distplot(df["Absenteeism time in hours"])
plt.title("Distribution of Absenteeism time in hours")
plt.show()

Conclusion

The above plot confirms that the target variable Absenteeism time in hours is highly positively skewed.

Kurtosis
Kurtosis is the degree of peakedness of a distribution.

Data sets with high kurtosis tend to have a distinct peak near the mean, decline rather rapidly and have heavy tails.

Data sets with low kurtosis tend to have a flat top near the mean rather than a sharp peak.

Reference range for kurtosis
The reference standard is a normal distribution, which has a kurtosis of 3. Often, excess kurtosis is presented instead of kurtosis, where excess kurtosis is simply kurtosis - 3.

Mesokurtic curve

A normal distribution has kurtosis exactly 3 (excess kurtosis exactly 0). Any distribution with kurtosis ≈3 (excess ≈ 0) is called mesokurtic.

Platykurtic curve

A distribution with kurtosis < 3 (excess kurtosis < 0) is called platykurtic. As compared to a normal distribution, its central peak is lower and broader, and its tails are shorter and thinner.

Leptokurtic curve

A distribution with kurtosis > 3 (excess kurtosis > 0) is called leptokurtic. As compared to a normal distribution, its central peak is higher and sharper, and its tails are longer and fatter.

We can calculate kurtosis as follows:-

df['Absenteeism time in hours'].kurt()
38.77730707753998
Conclusion

The kurtosis value of the Absenteeism time in hours is much much greater than 3. So, we can conclude that the distribution curve is a Leptokurtic curve. Its central peak is higher and sharper and its tails are longer and fatter.

Distribution of target variable
Now, we should plot the distribution of target variable. We can use Seaborn's distplot() function to plot the distribution.

First, I will draw the plot using distplot() function. This function plot a univariate distribution of observations. This function combines the matplotlib hist() function with the seaborn kdeplot() function.

We can proceed as follows:-

y = df['Absenteeism time in hours']
plt.figure(figsize=(8,6))
sns.distplot(y, kde=False, fit=st.norm)
plt.title('Normal fit')
plt.show()

We can see that the data values do not fit the normal distribution well. So, I will change the fit to lognormal distribution.

plt.figure(figsize=(8,6))
sns.distplot(y, kde=False, fit=st.lognorm)
plt.title('Log Normal fit')
plt.show()

Conclusion

We can see that the Absenteeism time in hours data values follow the lognormal distribution relatively closely as compared to normal distribution.

Findings of univariate analysis
Findings of univariate analysis are as follows:-

• The target variable Absenteeism time in hours is highly positively skewed.

• Its distribution curve is a Leptokurtic curve. Its central peak is higher and sharper and its tails are longer and fatter.

• The Absenteeism time in hours data values follow the lognormal distribution relatively closely as compared to normal distribution.

12. Multivariate analysis
Examine relationship between target variable and categorical attributes
In the dataset, we have several categorical attributes like Seasons, Education, Social drinker and Social smoker. In this section, I will explore the relationship between these categorical attributes and target variable.

Frequency distribution and visualization of categorical attributes
Seasons is a categorical attribute. We can find out what categories exist and how many values belong to each category using the value_counts() method as follows:-

df['Seasons'].value_counts()
4    195
2    192
3    183
1    170
Name: Seasons, dtype: int64
df['Seasons'].value_counts().plot(kind = 'bar', figsize=(10,5))
plt.title('Absenteeism time in hours in various seasons')
plt.xlabel('Seasons')
plt.ylabel('Absenteeism time in hours')
plt.legend()
plt.show()

Conclusion

Seasons attribute contain 4 data values as 1, 2, 3 and 4. These values represent 4 different seasons in a year which are coded as summer = 1, autumn = 2, winter = 3, spring = 4. So, we can conclude that spring contains highest number of Absenteeism time in hours.

Similarly, Education, Social drinker and Social smoker are also categorical attributes. We can visualize their frequency distribution using the value_counts() method and visualize them as follows:-

df['Education'].value_counts()
1.0    611
3.0     79
2.0     46
4.0      4
Name: Education, dtype: int64
df['Education'].value_counts().plot(kind = 'bar', figsize=(10,5))
plt.title('Absenteeism time in hours in various seasons')
plt.xlabel('Education')
plt.ylabel('Absenteeism time in hours')
plt.legend()
plt.show()

Conclusion

Education categorical attribute is coded as 1.0, 2.0, 3.0 and 4.0 which stands for different categories. The categories are high school (1), graduate (2), postgraduate (3), master and doctor (4). We can see that the high school category consists of highest number of Absenteeism time in hours.

df['Social drinker'].value_counts()
1    420
0    320
Name: Social drinker, dtype: int64
df['Social drinker'].value_counts().plot(kind = 'bar', figsize=(10,5))
plt.title('Absenteeism time in hours in various seasons')
plt.xlabel('Social drinker')
plt.ylabel('Absenteeism time in hours')
plt.legend()
plt.show()

Conclusion

Social drinker consists of two categories - (yes=1; no=0). From the graph, we can conclude that Social drinker have higher number of Absenteeism time in hours.

df['Social smoker'].value_counts()
0    686
1     54
Name: Social smoker, dtype: int64
df['Social smoker'].value_counts().plot(kind = 'bar', figsize=(10,5))
plt.title('Absenteeism time in hours in various seasons')
plt.xlabel('Social smoker')
plt.ylabel('Absenteeism time in hours')
plt.legend()
plt.show()

Conclusion

Social smoker consists of two categories - (yes=1; no=0). From the graph, we can conclude that Social smoker have lesser number of Absenteeism time in hours.

Findings of multivariate analysis
Findings of bivariate analysis are as follows:-

• The spring season contains highest number of Absenteeism time in hours.

• The high school category consists of highest number of Absenteeism time in hours.

• The Social drinker category have higher number of Absenteeism time in hours.

• The Social smoker category have lesser number of Absenteeism time in hours.

Estimating correlation coefficients
Our dataset is very small. So, we can compute the standard correlation coefficient (also called Pearson's r) between every pair of attributes. We can compute it using the df.corr() method as follows:-

correlation = df.corr()
Our target variable is Absenteeism time in hours. So, we should check how each attribute correlates with the Absenteeism time
in hours variable. We can do it as follows:-

correlation['Absenteeism time in hours'].sort_values(ascending=False)
Absenteeism time in hours          1.000000
Height                             0.144420
Son                                0.113756
Age                                0.065760
Social drinker                     0.065067
Transportation expense             0.027585
Hit target                         0.026695
Work load Average/day              0.024749
Month of absence                   0.023779
Service time                       0.019029
Weight                             0.015789
Seasons                           -0.005615
Social smoker                     -0.008936
Education                         -0.046235
Body mass index                   -0.049719
Distance from Residence to Work   -0.088363
Disciplinary failure              -0.124248
Day of the week                   -0.124361
Reason for absence                -0.173116
Name: Absenteeism time in hours, dtype: float64
Interpretation of correlation coefficient

The correlation coefficient ranges from -1 to +1.

When it is close to +1, this signifies that there is a strong positive correlation. So, we can see that there is a small positive correlation between Absenteeism time in hours and Height.

When it is clsoe to -1, it means that there is a strong negative correlation. So, there is a small negative correlation between Absenteeism time in hours and Reason for absence.

When it is close to 0, it means that there is no correlation. So, there is no correlation between Absenteeism time in hours and Seasons.

Discover patterns and relationships
An important step in EDA is to discover patterns and relationhsips between variables in the dataset. We will use the following graphs and plots to explore the patterns and relationships in the dataset.

Correlation Heat Map
plt.figure(figsize=(16,12))
plt.title('Correlation of Attributes with Absenteeism time in hours')
a = sns.heatmap(correlation, square=True, annot=True, fmt='.2f', linecolor='white')
a.set_xticklabels(a.get_xticklabels(), rotation=90)
a.set_yticklabels(a.get_yticklabels(), rotation=30)           
plt.show()

Conclusion

From the above correlation heat map, we can conclude that :-

Month of absence and Seasons are positively correlated (correlation coefficient = 0.41).

Body mass index and Service time are positively correlated (correlation coefficient = 0.50).

Smilarly, Body mass index and Age are positively correlated (correlation coefficient = 0.47).

Also, Body mass index and Weight are highly positively correlated (correlation coefficient = 0.90).

Pair Plot
num_var = ['Transportation expense', 'Distance from Residence to Work', 'Service time', 'Age', 'Work load Average/day ', 
           'Absenteeism time in hours']
sns.pairplot(df[num_var], kind='scatter', diag_kind='hist')
plt.show()

Conclusion

The above pair plot confirms that there is strong positive correlation between Service time and Age.

Similarly, Distance from Residence to Work and Transportation expense are positively correlated.

Scatter Plot of Absenteeism time in hours and height
sns.lmplot(x='Absenteeism time in hours', y='Height', data=df)
plt.show()

Conclusion

The above scatter-plot shows that there is a mildly positive correlation between Absenteeism time in hours and Height. Majority of data values lie below the fitted regression line.

Scatter Plot of Absenteeism time in hours and son
sns.lmplot(x='Absenteeism time in hours', y='Son', data=df)
plt.show()

Conclusion

The above scatter-plot shows that there is a weak correlation between Absenteeism time in hours and Son.

Scatter Plot of Body Mass Index and Service time
sns.lmplot(x='Body mass index', y='Service time', data=df)
plt.show()

Conclusion

The above scatter-plot shows that there is a strong positive correlation between Body mass index and Service time. Approximately, half of the data values lie below the fitted regression line and half of the values lie above it.

Box Plot of Absenteeism time in hours and age
plt.rcParams['figure.figsize']=(15,5)
ax = sns.boxplot(x='Age', y='Absenteeism time in hours', data=df)

Conclusion

The above box-plot confirms that the people aged 34 or 58 have highest number of Absenteeism time in hours.

plt.rcParams['figure.figsize']=(15,5)
ax = sns.boxplot(x='Son', y='Absenteeism time in hours', data=df)

Conclusion

The people who have 2 sons have highest number of Absenteeism time in hours.

Findings of multivariate analysis
Findings of multivariate analysis are as follows:-

• Month of absence and Seasons are positively correlated (correlation coefficient = 0.41).

• Body mass index and Service time are positively correlated (correlation coefficient = 0.50).

• Smilarly, Body mass index and Age are positively correlated (correlation coefficient = 0.47).

• Also, Body mass index and Weight are highly positively correlated (correlation coefficient = 0.90).

• The pair plot confirms that there is strong positive correlation between Service time and Age.

• Similarly, Distance from Residence to Work and Transportation expense are positively correlated.

• There is a mildly positive correlation between Absenteeism time in hours and Height. Majority of data values lie below the fitted regression line.

• There is a weak correlation between Absenteeism time in hours and Son.

• There is a strong positive correlation between Body mass index and Service time. Approximately, half of the data values lie below the fitted regression line and half of the values lie above it.

• The people aged 34 or 58 have highest number of Absenteeism time in hours.

• The people who have 2 sons have highest number of Absenteeism time in hours.
