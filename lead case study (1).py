#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE

from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import precision_recall_curve



# 

# # Import data

# In[2]:


lead_df=pd.read_csv("Leads.csv")
pd.set_option('display.max_columns',None)
lead_df.head()


# # Inspecting the dataframe

# In[3]:


lead_df.info()


# In[4]:


lead_df.describe()


# In[5]:


#Missing data percentage
round(lead_df.isnull().sum()/lead_df.shape[0],2)


# # Data preparation
# converting 1/0

# In[6]:


#Encoding the variable with yes/no
for feature in ['Do Not Email','Do Not Call','Search','Magazine','Newspaper Article','X Education Forums','Newspaper','Digital Advertisement',
                'Through Recommendations','Receive More Updates About Our Courses','Update me on Supply Chain Content','Get updates on DM Content'
               ,'I agree to pay the amount through cheque','A free copy of Mastering The Interview']:
        lead_df[feature]=lead_df[feature].apply(lambda x : 1 if x =='yes'else 0)  
                                                
lead_df.head()


# In[7]:


#Listing the categorial variable yet to be encoded
lead_df.select_dtypes(include='object').info()


# In[8]:


#Checking the labels for remaining categorial columns
for col in lead_df.iloc[:,1:].select_dtypes(include='object').columns:
    print (col)
    print("                                                                                ")
    print(lead_df[col].value_counts(normalize=True))
    print("                                                                                ")
    


# In[9]:


#Converting all selects to NaN
lead_df=lead_df.replace('Select',np.nan)


# In[10]:


lead_df.info()


# # Missing value handling

# In[11]:


#Dropping columns having null values more  than 70%
lead_df=lead_df.drop(lead_df.loc[:,list(round(lead_df.isnull().sum()/lead_df.shape[0],2)>0.70)].columns,1)


# In[12]:


#replace from nan to 'Not sure'
lead_df['Lead Quality']=lead_df['Lead Quality'].replace(np.nan,'Not Sure')


# In[13]:


lead_df=lead_df.drop(['Asymmetrique Activity Index','Asymmetrique Activity Score','Asymmetrique Profile Index','Asymmetrique Profile Score'
                     ],1)


# In[14]:


#We can input the Mumbai into all the nulls as most of the null values belongs to Mumbai
lead_df['City']=lead_df['City'].replace(np.nan,'Mumbai')

#Input nan in 3 specialisation sections
lead_df['Specialization']=lead_df['Specialization'].replace(np.nan,'Other_Specialization')

#Input null values in the tag column having more than 30% of 'will revert after reading the email'
lead_df['Tags']=lead_df['Tags'].replace(np.nan,'Will revert after reading the email')

#Input null values for more than 99% is of "Better career prospects"
lead_df['What matters most to you in choosing a course']=lead_df['What matters most to you in choosing a course'].replace(np.nan,'Better career prospects')

# Input null value having 85% data in Unemployed section
lead_df['What is your current occupation']=lead_df['What is your current occupation'].replace(np.nan,'Unemployed')

# Input null value having 95% data in country section in 'India'
lead_df['Country']=lead_df['Country'].replace(np.nan,'India')



# In[15]:


# Checking missing data percentage in updated dataframe
round(100*(lead_df.isnull().sum()/len(lead_df.index)),2)


# In[16]:


# Remaining null values are less than 2% and then can directly dropped
lead_df.dropna(inplace=True)
lead_df.head()


# # EDA

# In[17]:


#Start with a target variable if any data imbalace
lead_df['Converted'].value_counts(normalize=True)


# In[18]:


for i, feature in enumerate(['Lead Source', 'Lead Origin']):
    plt.subplot(1, 2, i+1)
    sns.countplot(data=lead_df, x=feature, hue='Converted')
    plt.xticks(rotation=90)
    plt.xlabel(feature)
    plt.ylabel('Count')
    plt.title(f'Count Plot of {feature} vs. Converted')

# Show the plots
plt.show()


# In[19]:


# Count of leads from various sources are close to negligible so to put into 'others'section
lead_df['Lead Source']=lead_df['Lead Source'].replace(['Click2call','Live Chat','NC_EDM','Pay per Click Ads','Press_Release'
        ,'Social Media','welearn','bing','blog','testone','welearnblog_Home','youtubechannel'],'Other_Lead Source')

lead_df['Lead Source']=lead_df['Lead Source'].replace("google",'Google')


# In[20]:


#Plotting lead source again
sns.countplot(x='Lead Source',hue='Converted',data=lead_df)
plt.xticks(rotation='vertical')
plt.show()


# In[21]:


# Set the figure size
fig=plt.subplots(figsize=(6,6))

for i, feature in enumerate(["TotalVisits", "Total Time Spent on Website"]):
    plt.subplot(2, 2, i+1)
    plt.subplots_adjust(hspace=2.0)
    sns.boxplot(lead_df[feature])
    plt.tight_layout()
    
plt.show()


# In[22]:


# many outliers in the totalvisits so we cap to 95 percentile
q1=lead_df["TotalVisits"].quantile(0.95)
lead_df["TotalVisits"][lead_df["TotalVisits"] >=q1] =q1


# In[23]:


fig=plt.subplots(figsize=(6,6))

for i, feature in enumerate(["TotalVisits", "Total Time Spent on Website"]):
    plt.subplot(2, 2, i+1)
    plt.subplots_adjust(hspace=2.0)
    sns.boxplot(y=feature, x='Converted',data=lead_df)
    plt.tight_layout()
    
plt.show()


# In[27]:


plt.figure(figsize=(12, 6))
sns.countplot(x="Last Activity", hue="Converted", data=lead_df)
plt.xticks(rotation=90)  # Rotate x-axis labels vertically for readability
plt.xlabel("Last Activity")
plt.ylabel("Count")
plt.title("Count Plot of Last Activity vs. Converted")
plt.legend(title="Converted", loc='upper right')
plt.show()


# In[24]:


# Converting all low count  category into other section

lead_df['Last Activity']=lead_df['Last Activity'].replace(['Had a Phone Conversation','View in browser link Clicked','Visited Booth in Tradeshow'
            ,'Approached upfront','Resubscribed to emails','Email Received','Email Marked Spam'],'Other Activity')

#Plot the last activity again

sns.countplot(x="Last Activity", hue="Converted", data=lead_df)
plt.xticks(rotation='vertical')
plt.show()


# In[25]:


# Set the figure size
fig=plt.subplots(figsize=(10,6))

for i, feature in enumerate(["Specialization"]):
    plt.subplot(2, 2, i+1)
    plt.subplots_adjust(hspace=2.0)
    sns.countplot (x=feature, hue="Converted", data=lead_df)
    plt.xticks(rotation='vertical')
    plt.tight_layout()
plt.show()    


    


# In[26]:


lead_df['What is your current occupation']=lead_df['What is your current occupation'].replace("Other",'Other_Occupation')


# In[27]:


lead_df[["Search","Magazine","Newspaper Article","X Education Forums","Newspaper","Digital Advertisement","Through Recommendations"
        ,"Update me on Supply Chain Content","Get updates on DM Content","I agree to pay the amount through cheque",
         "A free copy of Mastering The Interview"]].describe()


# In[28]:


# Set the figure size
fig=plt.subplots(figsize=(10,10))

for i, feature in enumerate(["Lead Quality","Tags"]):
    plt.subplot(2, 2, i+1)
    plt.subplots_adjust(hspace=2.0)
    sns.countplot (x=feature, hue="Converted", data=lead_df)
    plt.xticks(rotation='vertical')
    plt.tight_layout()
  


# In[29]:


#Converting to all low count category to other category
lead_df['Tags']=lead_df['Tags'].replace(['In confusion whether part time or DLP','in touch with EINS','Diploma holder (Not Eligible)'
 ,'Approched upfront','Graduation in progress','number not provided','opp hangup','Still Thinking','Lost to Others',
'Shall take in the next coming month','Lateral student','Interested in Next batch','Recognition issue(DEC approval)',
'Want to take admission but has financial problems','University not recognized'],'Other_Tags')

#Plot the Tags again
sns.countplot (x="Tags", hue="Converted", data=lead_df)
plt.xticks(rotation='vertical')
plt.show()


# In[30]:


# Dropping unnecessary columns
lead_df=lead_df.drop(['Lead Number','What matters most to you in choosing a course','Search','Magazine','Newspaper Article',
                        
'X Education Forums','Newspaper','Digital Advertisement','Through Recommendations','Receive More Updates About Our Courses',
                        
'Update me on Supply Chain Content','Get updates on DM Content','I agree to pay the amount through cheque',
                        
'A free copy of Mastering The Interview','Country','Do Not Call'],1)

lead_df.head()


# # Dummy variable creation

# In[31]:


dummy=pd.get_dummies(lead_df[['Lead Origin','Lead Source','Last Activity','Specialization','What is your current occupation',
                              'Tags','Lead Quality','City','Last Notable Activity']], drop_first=True)
dummy.head()


# In[32]:


lead_df=lead_df.drop(['Lead Origin','Lead Source','Last Activity','Specialization','What is your current occupation',
                              'Tags','Lead Quality','City','Last Notable Activity'],axis=1)
lead_df.head()


# In[33]:


lead_df=pd.concat([lead_df,dummy],axis=1)
lead_df.head()


# # Test train split
# 

# In[34]:


#putting feature variable to X
x=lead_df.drop(['Prospect ID', 'Converted'],axis=1)
#putting response vaqriable to y
y=lead_df['Converted']
print(y)
x.head()


# In[35]:


#splitting the date into train and test
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.7,test_size=0.3,random_state=100)


# # Feature scaling

# In[36]:


scaler=StandardScaler()
x_train[['TotalVisits','Total Time Spent on Website','Page Views Per Visit']]=scaler.fit_transform(x_train[['TotalVisits','Total Time Spent on Website','Page Views Per Visit']])
x_train.head()


# In[37]:


# Checking the conversion rate
print("Conversion rate is ",(sum(lead_df['Converted'])/len(lead_df['Converted'].index))*100)


# # Looking correlations

# In[38]:


#Correlations between different numerical variables for both the converted and non-converted cases
conv_corr=lead_df.corr()

#Correlations matric to find out top correlations
conv_corr_unstacked=conv_corr.unstack().sort_values(kind="quicksort")
conv_corr.where(np.triu(np.ones(conv_corr.shape),k=1).astype(np.bool)).stack().sort_values(ascending=False).head(10)


# In[39]:


#Dropping highly correlated feature
x_test=x_test.drop(['Lead Source_Facebook','Last Notable Activity_Unsubscribed','Last Notable Activity_SMS Sent',
'Last Notable Activity_Email Opened','Last Notable Activity_Unreachable','Last Notable Activity_Email Link Clicked',
'Last Notable Activity_Page Visited on Website'],1)

x_train=x_train.drop(['Lead Source_Facebook','Last Notable Activity_Unsubscribed','Last Notable Activity_SMS Sent',
'Last Notable Activity_Email Opened','Last Notable Activity_Unreachable','Last Notable Activity_Email Link Clicked',
'Last Notable Activity_Page Visited on Website'],1)


# In[40]:


conv_corr=x_train.corr()


# In[41]:


conv_corr.where(np.triu(np.ones(conv_corr.shape),k=1).astype(np.bool)).stack().sort_values(ascending=False).head(10)


# # Model Building

# In[42]:


logm1=sm.GLM(y_train,(sm.add_constant(x_train)),family=sm.families.Binomial())
logm1.fit().summary()


# # Feature selection using RFE

# In[43]:


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()


# In[51]:


from sklearn.feature_selection import RFE
rfe = RFE(logreg)     # running RFE with 15 variable as output
rfe = rfe.fit(x_train,y_train)


# In[52]:


rfe.support_


# In[53]:


list(zip(x_train.columns, rfe.support_, rfe.ranking_))


# In[54]:


col=x_train.columns[rfe.support_]


# In[55]:


#Assessing the model with stats model
x_train_sm=sm.add_constant(x_train[col])
logm2=sm.GLM(y_train,x_train_sm,family=sm.families.Binomial())
res=logm2.fit()
res.summary()


# In[56]:


# Getting the predicted values on the train set
y_train_pred=res.predict(x_train_sm)
y_train_pred[:10]


# In[57]:


y_train_pred=y_train_pred.values.reshape(-1)
y_train_pred[:10]


# # creating a data frame with true conversion status

# In[58]:


y_train_pred_final=pd.DataFrame({'Convert':y_train.values,'Convert_Prob':y_train_pred})
y_train_pred_final['Pros_ID']=y_train.index
y_train_pred_final.head()


# In[59]:


# Creating a new column 'predicted' with 1 if convert_prob>0.5 else 0
y_train_pred_final['predicted']=y_train_pred_final.Convert_Prob.map(lambda x:1 if x>0.5 else 0)
y_train_pred_final.head()


# In[60]:


print("Accuracy score",metrics.accuracy_score(y_train_pred_final.Convert,y_train_pred_final.predicted))


# # creating VIFs

# In[62]:


def calculate_vif(x_train):
    vif_df=pd.DataFrame()
    vif_df['Features']=x_train.columns
    vif_df['Variance Inflation Factor']=[variance_inflation_factor(x_train.values,i)for i in range(x_train.shape[1])]
    vif_df['Variance Inflation Factor']=round(vif_df['Variance Inflation Factor'],2)
    vif_df=vif_df.sort_values(by='Variance Inflation Factor',ascending=False)
    print(vif_df)
    
calculate_vif(x_train[col])


# In[63]:


col=col.drop('Tags_invalid number')
col


# In[64]:


# re-run the model using the selected variable
x_train_sm=sm.add_constant(x_train[col])
logm=sm.GLM(y_train,x_train_sm,family=sm.families.Binomial())
res=logm.fit()
res.summary()


# In[65]:


y_train_pred=res.predict(x_train_sm).values.reshape(-1)
y_train_pred_final['Covert_Prob']=y_train_pred

#creating new column 'predicted' with 1 if convert_prob>0.5 else 0
y_train_pred_final['predicted']=y_train_pred_final.Convert_Prob.map(lambda x:1 if x>0.5 else 0)
y_train_pred_final.head()


# In[66]:


# Check the overall accuracy
print("Accuracy score",metrics.accuracy_score(y_train_pred_final.Convert,y_train_pred_final.predicted))


# In[67]:


# Check the vif's again
calculate_vif(x_train[col])


# In[73]:


# Function name : evalute model
# argumet :y_true,y_predicted
# prints confusion matrix, accuracy,sensitivity,specificity,false positive rate,positive prediction value
# return accuracy,sensitivity,specificity.

def evalute_model(y_true,y_predicted,print_score=False):
    confusion=metrics.confusion_matrix(y_true,y_predicted)
    #predicted        non-converted            converted
    #Actual
    #non-converted         TN                     FP
    #converted             TN                     FP
    
    
    TP=confusion[1,1]#true positive
    TN=confusion[0,0]#true negative
    FP=confusion[0,1]#false positive
    FN=confusion[1,0]#false negative
    
    
    accuracy_sc=metrics.accuracy_score(y_true,y_predicted)
    sensitivity_score=TP/float(TP+FN)
    specificity_score=TN/float(TN+FP)
    precision_sc=precision_score(y_true,y_predicted)
    
    
    if print_score:
        print ("Confusion Matrix:\n",confusion)
        print ("Accuracy:",accuracy_sc)
        print ("Sensitivity:",sensitivity_score)
        print ("Specificity:",specificity_score)
        print ("Precision:",precision_sc)
        
    return accuracy_sc, sensitivity_score, specificity_score, precision_sc
    


# In[74]:


#Evaluting model
evalute_model(y_train_pred_final.Convert,y_train_pred_final.predicted,print_score=True)


# In[75]:


def draw_roc(actual,probs):
    fpr,tpr,thresholds=metrics.roc_curve(actual,probs,drop_intermediate=False)
    auc_score = metrics.roc_auc_score( actual, probs )
    plt.figure(figsize=(5, 5))
    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    return None
    


# In[79]:


fpr, tpr, thresholds = metrics.roc_curve( y_train_pred_final.Convert, y_train_pred_final.Convert_Prob, drop_intermediate = False )


# In[80]:


draw_roc(y_train_pred_final.Convert, y_train_pred_final.Convert_Prob)


# # Finding optimal value of the cut off

# In[82]:


# Let's create columns with different probability cutoffs 
numbers = [float(x)/10 for x in range(10)]
for i in numbers:
    y_train_pred_final[i]= y_train_pred_final.Convert_Prob.map(lambda x: 1 if x > i else 0)
y_train_pred_final.head()


# In[84]:


# Now let's calculate accuracy sensitivity and specificity for various probability cutoffs.
cutoff_df = pd.DataFrame( columns = ['prob','accuracy','sensi','speci'])
from sklearn.metrics import confusion_matrix

# TP = confusion[1,1] # true positive 
# TN = confusion[0,0] # true negatives
# FP = confusion[0,1] # false positives
# FN = confusion[1,0] # false negatives

num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
for i in num:
    cm1 = metrics.confusion_matrix(y_train_pred_final.Convert, y_train_pred_final[i] )
    total1=sum(sum(cm1))
    accuracy = (cm1[0,0]+cm1[1,1])/total1
    
    speci = cm1[0,0]/(cm1[0,0]+cm1[0,1])
    sensi = cm1[1,1]/(cm1[1,0]+cm1[1,1])
    cutoff_df.loc[i] =[ i ,accuracy,sensi,speci]
print(cutoff_df)


# In[85]:


# Let's plot accuracy sensitivity and specificity for various probabilities.
cutoff_df.plot.line(x='prob', y=['accuracy','sensi','speci'])
plt.show()


# In[87]:


## From the curve above, 0.2 is the optimum point to take it as a cutoff probability.

y_train_pred_final['final_predicted'] = y_train_pred_final.Convert_Prob.map( lambda x: 1 if x > 0.2 else 0)

y_train_pred_final.head()


# # Precision trade cut off

# In[89]:


p, r, thresholds = precision_recall_curve(y_train_pred_final.Convert, y_train_pred_final.Convert_Prob)
plt.plot(thresholds, p[:-1], "g-")
plt.plot(thresholds, r[:-1], "r-")
plt.show()


# In[90]:


y_train_pred_final=y_train_pred_final.iloc[:, :3]
y_train_pred_final['predicted']=y_train_pred_final.Convert_Prob.map(lambda x:1 if x>0.27 else 0)
y_train_pred_final['Lead_Score']=y_train_pred_final.Convert_Prob.map(lambda x:round(x*100))
y_train_pred_final.head()


# In[91]:


#Evaluting modelperformance on training data
evalute_model(y_train_pred_final.Convert,y_train_pred_final.predicted,print_score=True)


# In[95]:


# Getting the predicted value on train set
x_test_sm=sm.add_constant(x_test[col])
y_test_pred=res.predict(x_test_sm)

y_test_df=pd.DataFrame(y_test)
y_test_pred_df=pd.DataFrame(y_test_pred, columns=["Converting_Probability"])
y_test_df['Prospect ID']=y_test_df.index

y_predicted_final=pd.concat([y_test_df.reset_index(drop=True),y_test_pred_df.reset_index(drop=True)],axis=1)
y_predicted_final['final_predicted']=y_predicted_final.Converting_Probability.map(lambda x:1 if x>0.27 else 0)
y_predicted_final['Lead_Score']=y_predicted_final.Converting_Probability.map(lambda x:round(x*100))
y_predicted_final.head()


# In[96]:


#Evaluting modelperformance on test data
evalute_model(y_predicted_final.Converted,y_predicted_final.final_predicted,print_score=True)


# # Final Model

# In[105]:


#Build a logistic regression model and returns predicted values on training dataset
# when training data,test data and probability cutoff is given

def build_model_cutoff(x_train,y_train,x_test,y_text,cutoff=0.5):
    
    #Train model
    x_train_sm=sm.add_constant(x_train)
    logm=sm.GLM(y_train,x_train_sm,family=sm.families.Binomial())
    res=logm.fit()
    
    y_train_pred=res_predict(x_train_sm).values.reshape(-1)
    
    
    y_train_pred_final=pd.DataFrame({'Prospect ID':y_train.index,'Converted':y_train.values,'Convert_Probability':y_train_pred})
    y_train_pred_final['Convert_predicted']=y_train_pred_final.Convert_Probability.map(lambda x:1 if x> cutoff else 0)
    y_train_pred_final['Lead_Score']=y_train_pred_final.Convert_Probability.map(lambda x:round(x*100))
    print("-------------------------Result of training data-------------------")
    print(y_train_pred_final.head())

    
   #Predicting lead score on test data
    x_test.sm=sm.add_constant(x_test)
    y_test_pred=res.predict(x_test_sm)

    y_test_pred_final=pd.DataFrame({'Prospect ID':y_test.index,'Converted':y_test.values,'Convert_Probability':y_test_pred})
    y_test_pred_final['Convert_predicted']=y_test_pred_final.Convert_Probability.map(lambda x:1 if x> cutoff else 0)
    y_test_pred_final['Lead_Score']=y_test_pred_final.Convert_Probability.map(lambda x:round(x*100))
    y_test_pred_final.rest_index(inplace=True,drop=True)
    print("-------------------------Result of test data-------------------")
    print(y_test_pred_final.head())
    
    print("-------------------------Model Evaluation Metrics-------------------")
    evaluate_model(y_test_pred_final.Converted,y_test_pred_final.Convert_Predicted,print_score=True)
    
    
    return y_test_pred_final


# In[106]:


print("Features used in Final Model:",col)
print("-----------------------Feature Importance------------------")
print(res.params)


# In[ ]:




