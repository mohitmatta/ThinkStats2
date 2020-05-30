#!/usr/bin/env python
# coding: utf-8

# In[10]:


"""
MOHIT_MATTA_DSC530_FINAL_PROJECT.py
"""



get_ipython().run_line_magic('matplotlib', 'inline')

import thinkplot
import thinkstats2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import hypothesis
import statsmodels.formula.api as smf

# Read input advertising dataset 
advertisement_data=pd.read_csv('advertising.csv')

#Rename columns to remove spaces from names and use them in code 
advertisement_data=advertisement_data.rename(columns={"Daily Time Spent on Site": "Daily_Time_Spent"})
advertisement_data=advertisement_data.rename(columns={"Area Income": "Area_income"})
advertisement_data=advertisement_data.rename(columns={"Daily Internet Usage": "daily_internet_usage"})
advertisement_data=advertisement_data.rename(columns={"Clicked on Ad": "Clicked_on_Ad"})

male_ds=advertisement_data[advertisement_data.Male==1]
female_ds=advertisement_data[advertisement_data.Male==0]

#Print summary , information and sample records from dataset
print(advertisement_data.info())
print(advertisement_data.describe())
print(advertisement_data.head())


# Create a dataframe with clicked data and non clicked data
ad_data_clicked=advertisement_data[advertisement_data.Clicked_on_Ad == 1]
ad_data_nonclicked=advertisement_data[advertisement_data.Clicked_on_Ad == 0]


###############################################
############  Section 1- Histograms  ##########
###############################################

# Plot the histograms for Age
advertisement_data['Age'].plot.hist(bins=25)
plt.xlabel('Age in years')
thinkplot.show()

# Plot the histogram for Age variable with clicked data
ad_data_clicked['Age'].plot.hist(bins=25)
plt.xlabel('Age in years')
plt.title('Clicked data')
thinkplot.show()

# Plot the histogram for Age variable with non clicked data
ad_data_nonclicked['Age'].plot.hist(bins=25)
plt.xlabel('Non Clicked Data')
thinkplot.show()


#Plot histogram for Daily time Spent variable 
advertisement_data['Daily_Time_Spent'].plot.hist()
plt.xlabel('Daily time spent on website in minutes')
thinkplot.show()

#Plot histogram for Daily time Spent variable 
male_ds['Daily_Time_Spent'].plot.hist()
plt.xlabel('Daily time spent on website in minutes')
plt.title('Male daily time spent')
thinkplot.show()

#Plot histogram for Daily time Spent variable 
female_ds['Daily_Time_Spent'].plot.hist()
plt.xlabel('Daily time spent on website in minutes')
plt.title('Female Daily time spent')
thinkplot.show()

#Plot the histogram for Area Income variable
advertisement_data['Area_income'].plot.hist()
plt.xlabel('Area Income in Dollars')
thinkplot.show()

#Plot the histogram for Daily internet usage variable
advertisement_data['daily_internet_usage'].plot.hist()
plt.xlabel('Daily internet usage in minutes')
thinkplot.show()

#Create a dataset age range  to get the age categories
bins = [18, 30, 40, 50, 60, 70, 120]
labels = ['18-29', '30-39', '40-49', '50-59', '60-69', '70+']
advertisement_data['agerange'] = pd.cut(advertisement_data.Age, bins, labels = labels,include_lowest = True)

#Print the count of clicks as per age range
print(advertisement_data[advertisement_data['Clicked_on_Ad']==1]['agerange'].value_counts().head(10))



#Transform timestamp to create Hour,Day of week,month and Date columns 

advertisement_data['Timestamp']=pd.to_datetime(advertisement_data['Timestamp'])

#Now we shall introduce new columns Hour,Day of Week, Date, Month from timestamp
advertisement_data['Hour']=advertisement_data['Timestamp'].apply(lambda time : time.hour)
advertisement_data['DayofWeek'] = advertisement_data['Timestamp'].apply(lambda time : time.dayofweek)
advertisement_data['Month'] = advertisement_data['Timestamp'].apply(lambda time : time.month)
advertisement_data['Date'] = advertisement_data['Timestamp'].apply(lambda t : t.date())

#Top Ad clicked on specific date
print(advertisement_data[advertisement_data['Clicked_on_Ad']==1]['Date'].value_counts().head(5))


# Create histogram of month variable 
advertisement_data['Month'].plot.hist()
plt.xlabel('Month of year')
thinkplot.show()

#Create histogram of day variable
advertisement_data['DayofWeek'].plot.hist()
plt.xlabel('Day of Week')
thinkplot.show()




############################################################################
#############################Section 2 -PMF#################################
############################################################################


#Scenario 1 : Compare pmf of daily time spent for male vs female

male_ds=advertisement_data[advertisement_data.Male==1]
female_ds=advertisement_data[advertisement_data.Male==0]
male_pmf = thinkstats2.Pmf(male_ds.Daily_Time_Spent, label='male')
female_pmf = thinkstats2.Pmf(female_ds.Daily_Time_Spent, label='female')



#Plot pmf of daily time spent for male and female 
width=20
axis = [30, 90, 0, 0.01]
thinkplot.PrePlot(2)
#thinkplot.SubPlot(2)
thinkplot.Pmfs([male_pmf, female_pmf])
thinkplot.Config(xlabel='Daily Time Spent in Minutes', axis=axis)
thinkplot.show()


#Scenario 2: Compare pmf of daily time spent for age group 18-29 Vs 30-39

bins = [18, 30, 40, 50, 60, 70, 120]
labels = ['18-29', '30-39', '40-49', '50-59', '60-69', '70+']
advertisement_data['agerange'] = pd.cut(advertisement_data.Age, bins, labels = labels,include_lowest = True)

age_grp_30_to_39_ds=advertisement_data[advertisement_data.agerange=='30-39']
age_grp_18_to_29_ds=advertisement_data[advertisement_data.agerange=='18-29']

age_grp_30_to_39_pmf = thinkstats2.Pmf(age_grp_30_to_39_ds.Daily_Time_Spent, label='30-39')
age_grp_18_to_29_pmf = thinkstats2.Pmf(age_grp_18_to_29_ds.Daily_Time_Spent, label='18-29')




#Plot pmf of daily time spent for age group 18-29 and 30-39

width=5
axis = [40, 70, 0, 0.01]
thinkplot.PrePlot(2)
#thinkplot.SubPlot(2)
thinkplot.Pmfs([age_grp_30_to_39_pmf, age_grp_18_to_29_pmf])
thinkplot.Config(xlabel='Daily Time spent in minutes', axis=axis)
thinkplot.show()

#Scenario 3: Area income of people who clicked on Ad vs who did not click on Ad 


clicked_ds=advertisement_data[advertisement_data.Clicked_on_Ad==1]
nonclicked_ds=advertisement_data[advertisement_data.Clicked_on_Ad==0]

clicked_pmf = thinkstats2.Pmf(clicked_ds.Area_income, label='clicked')
nonclicked_pmf = thinkstats2.Pmf(nonclicked_ds.Area_income, label='nonclicked')

#Plot pmf of age range for clicked ads vs non clicked ads

width=1000
axis = [10000, 70000, 0, 0.01]
thinkplot.PrePlot(2)
#thinkplot.SubPlot(2)
thinkplot.Pmfs([clicked_pmf, nonclicked_pmf])
thinkplot.Config(xlabel='Area Income', axis=axis)
thinkplot.show()

############################################################################
#############################Section 3 -CDF#################################
############################################################################

age_grp_30_to_39_cdf = thinkstats2.Cdf(age_grp_30_to_39_ds.Daily_Time_Spent, label='30-39')
age_grp_18_to_29_cdf = thinkstats2.Cdf(age_grp_18_to_29_ds.Daily_Time_Spent, label='18-29')

thinkplot.PrePlot(2)
thinkplot.Cdfs([age_grp_30_to_39_cdf, age_grp_18_to_29_cdf])
thinkplot.Config(xlabel='Daily Time Spent in minutes', ylabel='CDF')
thinkplot.show()

male_cdf = thinkstats2.Cdf(male_ds.Daily_Time_Spent, label='male')
female_cdf = thinkstats2.Cdf(female_ds.Daily_Time_Spent, label='female')

thinkplot.PrePlot(2)
thinkplot.Cdfs([male_cdf, female_cdf])
thinkplot.Config(xlabel='Daily Time Spent in minutes', ylabel='CDF')
thinkplot.show()



##################################################################################################
############################# Section 4 -Analytical Distribution #################################
##################################################################################################


def MakeNormalModel(age):
    """Plots a CDF with a Normal model.

    age: sequence
    """
    cdf = thinkstats2.Cdf(age, label='variable')

    mean, var = thinkstats2.TrimmedMeanVar(age)
    std = np.sqrt(var)
    print('n, mean, std', len(age), mean, std)

    xmin = mean - 4 * std
    xmax = mean + 4 * std

    xs, ps = thinkstats2.RenderNormalCdf(mean, std, xmin, xmax)
    thinkplot.Plot(xs, ps, label='model', linewidth=4, color='0.8')
    thinkplot.Cdf(cdf)
    
    
MakeNormalModel(advertisement_data.Age)
thinkplot.Config(title='Age, linear scale', xlabel='Age in years',
                 ylabel='CDF', loc='upper right')
thinkplot.show()


MakeNormalModel(advertisement_data.Daily_Time_Spent)
thinkplot.Config(title='Daily_Time_Spent, linear scale', xlabel='Daily_Time_Spent in minutes',
                 ylabel='CDF', loc='upper right')
thinkplot.show()



MakeNormalModel(advertisement_data.Area_income)
thinkplot.Config(title='Area_income, linear scale', xlabel='Area income in dollars',
                 ylabel='CDF', loc='upper right')
thinkplot.show()


MakeNormalModel(advertisement_data.daily_internet_usage)
thinkplot.Config(title='daily_internet_usage, linear scale', xlabel='Daily internet usage in minutes',
                 ylabel='CDF', loc='upper right')
thinkplot.show()


def MakeNormalPlot(weights):
    """Generates a normal probability plot of birth weights.

    weights: sequence
    """
    mean, var = thinkstats2.TrimmedMeanVar(weights, p=0.01)
    std = np.sqrt(var)

    xs = [-5, 5]
    xs, ys = thinkstats2.FitLine(xs, mean, std)
    thinkplot.Plot(xs, ys, color='0.8', label='model')

    xs, ys = thinkstats2.NormalProbability(weights)
    thinkplot.Plot(xs, ys, label='variable')
    

    
MakeNormalPlot(advertisement_data.daily_internet_usage)
thinkplot.Config(title='daily_internet_usage, normal plot', xlabel='daily_internet_usage in minutes',
                 ylabel='CDF', loc='upper left')
thinkplot.show()


MakeNormalPlot(advertisement_data.Area_income)
thinkplot.Config(title='Area_income, normal plot', xlabel='CDF',
                 ylabel='area income in dollars', loc='upper left')
thinkplot.show()


log_Area_income = np.log10(advertisement_data.Area_income)
MakeNormalModel(log_Area_income)
thinkplot.Config(title='Area Income, log scale', xlabel='Area income (log10 kg)',
                 ylabel='CDF', loc='upper right')
thinkplot.show()


cdf = thinkstats2.Cdf(advertisement_data.Area_income, label='data')
cdf_log = thinkstats2.Cdf(np.log10(advertisement_data.Area_income), label='data')
xs, ys = thinkstats2.RenderParetoCdf(xmin=13996, alpha=2.5, 
                                     low=0, high=79484)

thinkplot.Plot(xs, 1-ys, label='model', color='0.8')


thinkplot.Cdf(cdf, complement=True) 
thinkplot.Config(xlabel='log10 household income',
                 ylabel='CCDF',
                 xscale='log',
                 yscale='log', 
                 loc='lower left')

thinkplot.show()



##################################################################################################
############################# Section 4 -Scatter Plots  #################################
##################################################################################################

#Scatter plot of Age vs Area Income
thinkplot.Scatter(advertisement_data.Age, advertisement_data.Area_income, alpha=1)
thinkplot.Config(xlabel='Age(years)',
                 ylabel='Area Income (Dollars)',
                 axis=[10, 70, 10000,100000],
                 legend=False)
thinkplot.show()

#Scatter plot of Age vs Daily internet usage
thinkplot.Scatter(advertisement_data.Age, advertisement_data.daily_internet_usage, alpha=1)
thinkplot.Config(xlabel='Age(years)',
                 ylabel='daily internet usage (in minutes)',
                 axis=[10, 70, 100,300],
                 legend=False)
thinkplot.show()


#Plot of Area_income vs Age with click hue 
sns.lmplot( x='Age', y='Area_income',data=advertisement_data, fit_reg=False, hue='Clicked_on_Ad', legend=True, palette="Blues")
thinkplot.show()

#Plot of daily_internet_usage vs Age with click hue 
sns.lmplot( x='Age', y='daily_internet_usage',data=advertisement_data, fit_reg=False, hue='Clicked_on_Ad', legend=True, palette="Reds")
thinkplot.show()

fig, axes = plt.subplots(figsize=(10, 6))
ax = sns.kdeplot(advertisement_data['Daily_Time_Spent'], advertisement_data['Age'], cmap="Reds", shade=True, shade_lowest=False)
ax = sns.kdeplot(advertisement_data['daily_internet_usage'],advertisement_data['Age'] ,cmap="Blues", shade=True, shade_lowest=False)
ax.set_xlabel('Time')
ax.text(20, 20, "Daily_Time_Spent", size=16, color='r')
ax.text(200, 60, "daily_internet_usage", size=16, color='b')
thinkplot.show()


plt.figure(figsize=(10,6))
sns.violinplot(x=advertisement_data['Male'],y=advertisement_data['Area_income'],data=advertisement_data,palette='viridis',hue='Clicked_on_Ad')
plt.title('Clicked on Ad distribution based on area distribution')
thinkplot.show()

##################################################################################################
############################# Section 4 -Covariance and Correlation  #################################
##################################################################################################

def Cov(xs, ys, meanx=None, meany=None):
    xs = np.asarray(xs)
    ys = np.asarray(ys)

    if meanx is None:
        meanx = np.mean(xs)
    if meany is None:
        meany = np.mean(ys)

    cov = np.dot(xs-meanx, ys-meany) / len(xs)
    return cov

print('\nCovariance coefficient between different variables is:\n')
print('Covariance between daily internet usage and age is',Cov( advertisement_data.daily_internet_usage,advertisement_data.Age))
print('Covariance between Area income and age is',Cov( advertisement_data.Area_income,advertisement_data.Age))
print('Covariance between daily time spent and age is',Cov(advertisement_data.Daily_Time_Spent,advertisement_data.Age))
print('Covariance between area income and daily internet usage is',Cov(advertisement_data.Area_income, advertisement_data.daily_internet_usage))
print('Covariance between daily internet usage and daily time spent is',Cov(advertisement_data.daily_internet_usage, advertisement_data.Daily_Time_Spent))
print('Covariance between area income and daily time spent is',Cov(advertisement_data.Area_income, advertisement_data.Daily_Time_Spent))

#Calculate Correlation coefficient 
print('\n Correlation coefficient calculated between different variables is:\n')
print(np.corrcoef( advertisement_data.daily_internet_usage,advertisement_data.Age))
print(np.corrcoef(advertisement_data.Area_income,advertisement_data.Age))
print(np.corrcoef( advertisement_data.Daily_Time_Spent,advertisement_data.Age))
print(np.corrcoef(advertisement_data.Area_income, advertisement_data.daily_internet_usage))
print(np.corrcoef(advertisement_data.daily_internet_usage, advertisement_data.Daily_Time_Spent))
print(np.corrcoef(advertisement_data.Area_income, advertisement_data.Daily_Time_Spent))

#Plot correlation cluster plot for different variables 
sns.clustermap(advertisement_data.corr())
thinkplot.show()

# Define function for Spearman correlation coeffeiceint
def SpearmanCorr(xs, ys):
    xranks = pd.Series(xs).rank()
    yranks = pd.Series(ys).rank()
    return Corr(xranks, yranks)

def Corr(xs, ys):
    xs = np.asarray(xs)
    ys = np.asarray(ys)

    meanx, varx = thinkstats2.MeanVar(xs)
    meany, vary = thinkstats2.MeanVar(ys)

    corr = Cov(xs, ys, meanx, meany) / np.sqrt(varx * vary)
    return corr

#Calculate Spearman Correlation coefficient 
print('\nSpearman correlation coefficient between different variables is:\n')
print(SpearmanCorr(advertisement_data.daily_internet_usage,advertisement_data.Age))
print(SpearmanCorr(advertisement_data.Area_income,advertisement_data.Age))
print(SpearmanCorr(advertisement_data.Daily_Time_Spent,advertisement_data.Age))
print(SpearmanCorr(advertisement_data.Area_income, advertisement_data.daily_internet_usage))
print(SpearmanCorr(advertisement_data.daily_internet_usage, advertisement_data.Daily_Time_Spent))
print(SpearmanCorr(advertisement_data.Area_income, advertisement_data.Daily_Time_Spent))


#Calculate correlation coefficient for non linear relationships  between log scales
print('\nCorrelation coefficient between different variables for non linear relationships is:\n')
print(Corr(advertisement_data.daily_internet_usage, np.log(advertisement_data.Age)))
print(Corr(np.log(advertisement_data.daily_internet_usage), advertisement_data.Age))
print(Corr(advertisement_data.Area_income, np.log(advertisement_data.Age)))
print(Corr(advertisement_data.Daily_Time_Spent, np.log(advertisement_data.Age)))
print(Corr(np.log(advertisement_data.Area_income), advertisement_data.Daily_Time_Spent))
print(Corr(advertisement_data.Area_income, np.log(advertisement_data.Daily_Time_Spent)))



##################################################################################################
############################# Section 5 -Hypothesis Testing  #####################################
##################################################################################################

#Run Hypothesis Testing 
class CorrelationPermute(thinkstats2.HypothesisTest):

    def TestStatistic(self, data):
        xs, ys = data
        test_stat = abs(thinkstats2.Corr(xs, ys))
        return test_stat

    def RunModel(self):
        xs, ys = self.data
        xs = np.random.permutation(xs)
        return xs, ys

class DiffMeansResample(hypothesis.DiffMeansPermute):
    """Tests a difference in means using resampling."""
    
    def RunModel(self):
        """Run the model of the null hypothesis.

        returns: simulated data
        """
        group1 = np.random.choice(self.pool, self.n, replace=True)
        group2 = np.random.choice(self.pool, self.m, replace=True)
        return group1, group2
    
def RunResampleTest(firsts, others):
    """Tests differences in means by resampling.

    firsts: DataFrame
    others: DataFrame
    """
    data = male_ds.Daily_Time_Spent.values, female_ds.Daily_Time_Spent.values
    ht = DiffMeansResample(data)
    p_value = ht.PValue(iters=10000)
    print('\nmeans permute total time spent on website')
    print('p-value =', p_value)
    print('actual =', ht.actual)
    print('ts max =', ht.MaxTestStat())

    data = (male_ds.daily_internet_usage.values,
            female_ds.daily_internet_usage.values)
    ht = hypothesis.DiffMeansPermute(data)
    p_value = ht.PValue(iters=10000)
    print('\nmeans permute daily internet usage minutes')
    print('p-value =', p_value)
    print('actual =', ht.actual)
    print('ts max =', ht.MaxTestStat())
    
    
class Totaltimespent(thinkstats2.HypothesisTest):

    def MakeModel(self):
        firsts, others = self.data
        self.n = len(firsts)
        self.pool = np.hstack((firsts, others))

        pmf = thinkstats2.Pmf(self.pool)
        self.values = range(35, 44)
        self.expected_probs = np.array(pmf.Probs(self.values))

    def RunModel(self):
        np.random.shuffle(self.pool)
        data = self.pool[:self.n], self.pool[self.n:]
        return data
    
    def TestStatistic(self, data):
        firsts, others = data
        stat = self.ChiSquared(firsts) + self.ChiSquared(others)
        return stat

    def ChiSquared(self, lengths):
        hist = thinkstats2.Hist(lengths)
        observed = np.array(hist.Freqs(self.values))
        expected = self.expected_probs * len(lengths)
        stat = sum((observed - expected)**2 / expected)
        return stat
    
class DiffMeansPermute(thinkstats2.HypothesisTest):

    def TestStatistic(self, data):
        group1, group2 = data
        test_stat = abs(group1.mean() - group2.mean())
        return test_stat

    def MakeModel(self):
        group1, group2 = self.data
        self.n, self.m = len(group1), len(group2)
        self.pool = np.hstack((group1, group2))

    def RunModel(self):
        np.random.shuffle(self.pool)
        data = self.pool[:self.n], self.pool[self.n:]
        return data

def RunTests(advertisement_data, iters=1000):
   
    n = len(advertisement_data)
    male_ds=advertisement_data[advertisement_data.Male==1]
    female_ds=advertisement_data[advertisement_data.Male==0]

    # compare time spent
    data = male_ds.Daily_Time_Spent.values, female_ds.Daily_Time_Spent.values
    ht = DiffMeansPermute(data)
    p1 = ht.PValue(iters=iters)

    data = (male_ds.daily_internet_usage.values,female_ds.daily_internet_usage.values)
    ht = DiffMeansPermute(data)
    p2 = ht.PValue(iters=iters)

    # test correlation
    advertisement_data=advertisement_data.rename(columns={"Daily Time Spent on Site": "Daily_Time_Spent"})
    advertisement_data=advertisement_data.rename(columns={"Daily Internet Usage": "daily_internet_usage"})
    data = advertisement_data.Daily_Time_Spent.values, advertisement_data.daily_internet_usage.values
    ht = CorrelationPermute(data)
    p3 = ht.PValue(iters=iters)
   
    # compare total time spent between males and females (chi-squared)
    data = male_ds.Daily_Time_Spent.values, female_ds.Daily_Time_Spent.values
    ht = Totaltimespent(data)
    p4 = ht.PValue(iters=iters)
    
    print('\nn\tp1\tp2\tp3\tp4' )
    print('%d\t%0.2f\t%0.2f\t%0.2f\t%0.2f' % (n, p1, p2, p3, p4))

    

RunResampleTest(male_ds, female_ds)
print('\nn-number of sample records\np1-p-value for total time spent on site\np2-p value for daily internet usage\np3-correlation pvalue \np4-Chi-squared p-value')
n = len(advertisement_data)
for _ in range(8):
    sample = thinkstats2.SampleRows(advertisement_data, n)
    RunTests(sample)
    n //= 2






#########################################################################################################
############################# Section 6 -Multiple Regression Model  #####################################
#########################################################################################################

         
# Build and run a regression model with one dependent variable and multiple explanatory variables

ad_data=pd.read_csv('advertising.csv')
ad_data=ad_data.rename(columns={"Daily Time Spent on Site": "Daily_Time_Spent"})
ad_data=ad_data.rename(columns={"Daily Internet Usage": "daily_internet_usage"})
ad_data=ad_data.rename(columns={"Area Income": "Area_Income"})
ad_data=ad_data.rename(columns={"Clicked on Ad": "Clicked_on_Ad"})

formula2 = 'Clicked_on_Ad ~ Daily_Time_Spent + Age + Area_Income + daily_internet_usage  '
results2 = smf.ols(formula2, data=ad_data).fit()
print(results2.summary())

   
formula = 'Clicked_on_Ad ~ Daily_Time_Spent + Age + Area_Income + daily_internet_usage + Male '
results = smf.ols(formula, data=ad_data).fit()
print(results.summary())
print(results.pvalues)
print(results.rsquared)

formula1 = 'Clicked_on_Ad ~ Daily_Time_Spent + Age + Area_Income<50000 + daily_internet_usage + Male==0'
results1 = smf.ols(formula1, data=ad_data).fit()
print(results1.summary())
print(results1.pvalues)
print(results1.rsquared)


prediction = results.predict(ad_data)
actual = ad_data['Clicked_on_Ad']


#Calculate accuracy of Regression model
from sklearn import metrics

print(metrics.mean_absolute_error(actual, prediction))
print(metrics.mean_squared_error(actual, prediction))
print(np.sqrt(metrics.mean_squared_error(actual, prediction)))
print(metrics.r2_score(actual, prediction))



# Build and run a regression model with one dependent variable and multiple explanatory variables

ad_data=pd.read_csv('advertising.csv')
ad_data=ad_data.rename(columns={"Daily Time Spent on Site": "Daily_Time_Spent"})
ad_data=ad_data.rename(columns={"Daily Internet Usage": "daily_internet_usage"})
ad_data=ad_data.rename(columns={"Area Income": "Area_Income"})
ad_data=ad_data.rename(columns={"Clicked on Ad": "Clicked_on_Ad"})

formula2 = 'Clicked_on_Ad ~ Daily_Time_Spent + Age + Area_Income + daily_internet_usage  '
results2 = smf.ols(formula2, data=ad_data).fit()
print(results2.summary())

   
formula = 'Clicked_on_Ad ~ Daily_Time_Spent + Age + Area_Income + daily_internet_usage + Male '
results = smf.ols(formula, data=ad_data).fit()

print(results.summary())
print('\np-values for variables are:\n')
print(results.pvalues)

print(results.rsquared)

formula1 = 'Clicked_on_Ad ~ Daily_Time_Spent + Age + Area_Income<50000 + daily_internet_usage + Male==0'
results1 = smf.ols(formula1, data=ad_data).fit()
print(results1.summary())
print(results1.pvalues)
print(results1.rsquared)


prediction = results.predict(ad_data)
actual = ad_data['Clicked_on_Ad']


#Calculate accuracy of Regression model
from sklearn import metrics

print(metrics.mean_absolute_error(actual, prediction))
print(metrics.mean_squared_error(actual, prediction))
print(np.sqrt(metrics.mean_squared_error(actual, prediction)))
print(metrics.r2_score(actual, prediction))


# In[ ]:





# In[ ]:





# In[ ]:




