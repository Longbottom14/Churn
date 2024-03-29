# **Customer Churn Prediction**

- Application: [Link](https://longbottom14-churn-streamlit-share-main-dpy8gy.streamlitapp.com/)

## **Background**

This project was inspired by [datatalkclub machine learning zoomcamp](https://youtube.com/playlist?list=PL3MmuxUbc_hIhxl5Ji8t4O6lPAOpHaCLR)

### **Importance**

Customer churn or attrition is the rate at which clients opt out of purchasing more of a company's products or services.

Customer churn analysis is a method of measuring this rate and churn prediction means detecting which customers are likely to leave a service or to cancel a subscription to a service

Customer churn is a critical prediction for many businesses because acquiring new clients often costs more than retaining existing ones

### **Churn Rate and Risk Ratio**

- **Global Churn Rate**

The churn rate, also known as rate of attrition, is the rate at which Customers stop doing business with an entity. It is most commonly expressed as the percentage of service subscribers who discontinue their subscriptions within a time period. For instance: Monthly churn rate refers to the percentage of customers lost over the course of a month.

**To calculate monthly churn rate** :

Divide the number of customers you lost over the month by the number of customers you had at the beginning of the month and multiply the result by hundred

- **Risk Ratio**

A measure of the risk of a certain event happening in one group compared to the risk of the same event happening in another group.

## **Objective Function**

Objective function is the function that it is desired to maximize or minimize.

Consider a business where every existing customer lifetime value is #5000 in profit when they purchase our services.

                    1 customer = #5000 ; customer lifetime value

The business started losing customers, and after looking into the situation, the business found out that customers were churning because the competition was offering promotions.

### - **Problem Definition**

We want to identify customers that show certain characteristics of churning, send them free coupons of #1000 to those customers and therefore reducing the global churn to 10%-15%.

To model this machine learning problem, we want to predict customers who are likely to churn, and send free coupons only to those customers.

### - **Metric**

We won't use fancy metrics like accuracy (imbalance data), f1 score (suitable for imbalance data), R0C_AUC and other defined metrics.

Stakeholders most times want to evaluate machine learning projects/models with regards to business objectives which are revenue and profits. From this article I got from my friend Salim. [link](https://towardsdatascience.com/predict-customer-churn-the-right-way-using-pycaret-8ba6541608ac)

I defined my custom metric as :

| Metric | Coupons (#) | Customer lifetime value(#) | Saved (#) |
| --- | --- | --- | --- |
| True Positive (TP) | -1000 | 5000 | 4000 |
| True Negative (TN) | 0 | 5000 | 5000 |
| False Positive (FP) | -1000 | 5000 | 4000 |
| False Negative (FN) | 0 | -5000 | -5000 |

Based on these custom metrics, the model was built to learn the business objective and solve the problem statements ; reduce customer churn and increase profit.

## **Notebooks**

- **Eda**
  - Numerical data Distributions,
  - Categorical data count plots ,
  - Correlation plots
  - Mutual Information plots
  - Churn rate and risk ratio

- **Model Training**
  - Base model I.e logistics regression
  - Ensembles and gradient boosting models
  - Cross Validations

- **Blending and Stacking**
  - Cross Validations
  - Final models
  - Model Blending
  - Exporting artifacts I.e saving the model
  - Model Explanation I.e Feature Importance with Eli5.

## **Deployment**

- Streamlit share application : [Https://Longbottom14-churn-streamlit-share-main-dpy8gy.streamlitapp.com](https://longbottom14-churn-streamlit-share-main-dpy8gy.streamlitapp.com/)

- Docker and flask: [https://github.com/Longbottom14/Churn/tree/main/deployment\_flask\_docker](https://github.com/Longbottom14/Churn/tree/main/deployment_flask_docker)

## **Data Analytics**

[https://github.com/Longbottom14/Churn/tree/main/dashboard](https://github.com/Longbottom14/Churn/tree/main/dashboard)

Interactive Microsoft Power BI dashboard to study data individual parts and extract useful information by identifying trends and patterns.

## **Conclusion**

For most businesses, making profit is a key business objective. Applying machine learning to this problem gave the following results:


| | Training data | Test data | 
| --- | --- | --- |
| Profit with machine learning model | #22,963,000 | #5,924,000 |
| Profit without machine learning model | #12,960,000 | #3,565,000 |
| Spillover | #10,003,000 | #2,359,000 |

## **Acknowledgement**

- Alexey Grigorev. Twitter: @AI\_grigor
- Salim Oyinlola @SalimOpines
