# Big-Data-power-consumption
prediction of electicity consumption in a household 

The Problem of Multivariate of Power Consumption Using Deep Learning
Big Data
M.M. Nekhubvi
1 Academy of Computer Science and Software Engineering
University of Johannesburg, South Africa
215000028@student.uj.ac.za
Abstract—Technological advancements have led to a growth in equipment of electricity -based, which has increase daily energy demand and excessive electricity consumption. The Energy consumption (EC) must be forecasted in order to improve power management and cooperation between electricity utilized in a building and smart grid. Power consumption forecast model-based household electricity usage and Long Short-Term Memory (LSTM)is suggested to efficiently reduce the waste of electricity resources in the generation of power and transmission plan. To remove the volatility of actual data on power use in this study, the sample data is first denoised based on household electricity. Then based on the samples that are pre-processed the LSTM model is used to train, and the model is verified and predict daily power consumption based on the power used in the household. The experimental outcome reveal that the prediction performance of the model performs better than the traditional Machine learning (ML) models.
Keywords-component; ML; LSTM; EC; forecast.
I.INTRODUCTION
Power consumption forecasting is a crucial technological strategy for managing power demand. An accurate power forecast model is required in order to rationally organize the power supply plan of the power grid and prevent the waste of power resources, which is considerable relevance for creating an energy-saving society [10]. The operation and planning of the electricity supply firm include the prediction of power usage. Regarding electricity supply and demand, reserve power must be ready for the steady supply of energy.
Recently, with the advancement of Deep Learning and Artificial Intelligence technologies, power supply businesses and researchers have developed a variety of approaches for estimating electricity demand [9]. However, because power is hard to store, it is essential to forecast demand. In this study, we present a LSTM neural network that can successfully forecast household energy usage by extracting spatial-temporal information. Long Short-term memory (LSTM) and deep neural network have been proven through experiments to be able to extract irregular aspects of electric power use [1]. The LSTM proposed to automatically forecast household use. Power consumption contain both spatial and temporal information and is multivariate time series. In order to predict the home power consumption, the LSTM neural network can extract the space-time feature from the power consumption variable [2].
In this study we will break down the paper into section to full understanding of how we will go about with the implementation of the prediction model. First, section we look at the problem statement where will look at the problem that we are trying to solve. Second, section will study the background/related work that are used to solve the power consumption using different AI methods and ML models to see the improvement from the two. Third section we propose how the system is going to be built. Then we show how the model results perform. Lastly, we conclude the paper.
II. PROBLEM STATEMENT
The problem of load shedding is a big issue in south Africa. So, researching of solution that can make a different towards solving the problem will help stop the issue of load shedding. The research topic will be looking at the how to predict power consumption using deep learning methods. This topic is a forecasting problem that is going to make use of deep learning methods LSTM to build the model that will help in prediction the usage of power consumption and see how the household electricity usage can be solved. The solution can help Eskom to be able to produce enough electricity to each household and this will solve the loadshedding problem.
III.BACKGROUND
Numerous research has been done to identify patterns in energy consumption data and forecast electrical energy usage to ensure a steady supply [3]. Three types of models are used to anticipate energy consumption: model based on machine learning, statistics, and deep learning. They are certain things that influence the household EC, Zhang ang Zhu [11] chose parameter such as income, population, temperature, living area and price as analytical indicators from numerous viewpoint that affect the power demand and supply. Their model was based on the particle swarm optimization (PSOs) to predict the EC data. The model used the support vector regression, and the model parameters are optimized with PSO, and cross-validation and this show better results in prediction accuracy.
Based on the time sequence and nonlinear characteristics of EC, Xu et al. [12] developed the LSTM model and combined with the CCM methods to study the dynamics of parameters between EC and temperature, precipitation, relative humidity, and wind speed. The model was used to anticipate urban EC and results shows that it is more accurate and has smaller annual forecast relative error than monthly forecast relative error.
According to three approaches, Table 1 presents the studies on prediction power usage.
Table 1
Related works on electric consumption prediction.
Category
Author
Methods
Description
Statistical based modeling
N. Funo [4], 2015
Linear regression
Electricity consumption Analysis of prediction performance according to time resolution
M.R. Braun et al. [5], 2014
Multiple regression
Electricity consumption using temperature and humidity records to improve performance
Machine learning based modeling
A. Bogomolov et al. [6], 2016
Random forest regression
Prediction energy consumption based on human dynamics analysis
R. K. Jain et al. [7], 2014
Support vector regression
Electricity consumption predictive performance according to temporal impact.
Deep learning-based modeling
K. Tea-young et al. [1], 2019
CNN-LSTM neural network
Prediction residential energy consumption
F. Liang et al. [8], 2019
LSTM and RNN
Research on the Innovation of Protecting Intangible Cultural
Heritage in the "Interne
However, multicollinearity issues might result from the connection between independent variable utilized for prediction. Another drawback of utilizing linear regression is that it might be challenging to find explanatory factors [4]. Same applies to ML as complex variable correlations or data volume grow, the current machine learning techniques suffer from significant overfitting. Long-term consumption is difficult to forecast when overfitting takes place [6]. So, the use deep learning method reduce noisy disturbance and randomness from extracting important features and power usage data. Even when there is large amount of existing data and complicated properties, these approaches can automatically extract and model key aspects [1]. The spatial-temporal aspects of the power usage, however, are challenging to model using traditional deep learning technique.
IV.PROPOSED METHOD
The study of overall architecture for forecasting residential power consumption used many techniques but the deep learning techniques show significant performance. The UCI machine learning repository dataset on electric power usage was used in this study [9]. The records on the data shows observations make up a multivariate dataset.
The right network topologies must be chosen to apply deep learning to our problem. in this instance, supervised learning is being used to produce accurate prediction. The technique the paper is going to examine is LSTM.
A. Dataset
The dataset for training the model was obtained in the https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption website due to it being hard to find ESKOM dataset for household power usage and the problem solved in the paper is household power usage. The data consists of data from 2006 to 2010 and the dataset shows the minute usage of electricity in a household. The set contains 2075259 rows of information between the years mentioned above. Now we will look at the details of the file containing the dataset and its parameters.
Household power usage file has 7 columns of parameter including: the power used by all appliances in the house, power that is not used but move around, the voltage of the power used, total power used in the household for an hour the sub-meters that are contained in the house. We will show the usage of electricity monthly, hourly, and yearly in the graph in the results section.
First, we read the file downloaded from the website above then we visualize the data for monthly, hourly, and daily to show the consistency of how the electricity is used on the household. The resampling of dataset is important due to the change of periodicity of the system and the dataset is very large. In order to determine the daily electricity use, the original data will thus be grouped by date in the next preparation stage. The information from the original dataset is displayed in section as follows:
Table 2. Datasets
Date
Time
Used power
Unused power
Voltage
Total power
16/12/2006
17:24
4.216
0.418
234.84
18.4
16/12/2006
17:25
5.36
0.436
233.63
23
16/12/2006
17:26
5.374
0.498
233740
23
..
..
..
..
..
..
B. Preprocessing
At this stage we prepare the dataset that we going to use to train the model and only certain parts of the data is going to be used. First, we clean the dataset and statistics of the data in minutes are changed into hours, days and monthly since the data was collected for four years then choose what we want the model we intend to use to train what we want to analyze. At this stage we can display the data of consumption per hour for the 4 years period, then monthly consumption for the period of the data then daily consumption.
For the features to create training for the model we used the time feature: monthly, hours, and yearly. National holiday features as a Boolean time series. Then we prepare the simple function to turn a time series into deep leaning ready dataset and our time series is supervised.
C. LSTM model
To provide long-term memory capacities, the Long Short-Term Memory model swaps out the hidden layers with LSTM gates components. The gate state, which permeates the whole hidden link, is the LSTMs strategic. Through the gate structure, LSTM regulates the state of the cell to add or remove information. The LSTM model can recall a lot of information because of the gate structure, which allows information to travel through only when it is specially needed. This model is similar to the RNN models, but the issue is that LSTM resolves the vanishing of the gradient by context of information. The Adam optimization algorithm is used to improve the LSTM model, this helps in updating the weight and the test set is used to evaluate the model performance.
The figure below shows the structure of the LSTM gates that it has:
Figure 1. LSTM gates [10]
The LSTM consists of three gates, which are: input gate, memory gate and the output gate. Both the previous time step’s hidden state and the current time step’s input are used to calculate the output of the long and short memory gates, which uses the sigmoid at the fully linked layer of the activation function [3]. From figure 1, Let it represent the input gate, ft represent the forget gate/ memory gate and ot represent the output gate. Ht-1 represents the LSTM output at previous time, Xt represents the data of the input layer at time t. And tanh represents the sigmoid and hyperbolic tangent activation function.
Now we will explain the mathematical form of how the working process LSTM model works.
The study will give the explanation of each equation in bolet form[4].
• Equation 1: identifies the information that need updating using input gate.
• Equation 2: The output gate calculation. The sigmoid function, which has a value between 0 and 1, is one of them. When its value is 0 , the gate is closed and when its 1 the gate is open.
• Equation 3: The forget gate can be used to ascertain the information state in the cell state. Then the forget gate monitor and, represents the memory cell’s value at time t-1 and outpts values ranging from 0 to 1 for each cell state. The estimate formula for the foget gate is: where 1 signies “total keep the information” and 0 means “entirely dispose of the information”.
• Equation 4: Determole the memory gate, choose the data to be remembered and then determine the provisional cell state.
• Equation 5: The ultimate outcome value of the simulation depends on the state of the memory cell at the present time Ct and the value of the output gate, assuming that is the final output of the LSTM unit at time t [10].
This section will help us develop the LSTM model using all the subsection mention above. next we will look at the results of the system and how the system perfom.
V. RESULTS
This part of the paper we are going to show and examine the outcome of the system. First, the display the result of the dataset visualization. The figure 2 shows the visualization of the daily mean resampling of power usage. Figure 4 shows the hourly usage, then the last figure shows the monthly mean of the resampling. Then next we will plot the correlation of them to show how the parameter correlates with each other. The are going to be three correlations including the hourly, daily and monthly.
Figure 2. Daily visualization
Figure 3. Houly visualization
Figure 4. Monthly visualization
figure 5. Hourly correlation
figure 6. Daily Corrrelation
Figure 7. monthly correlation
In the above figure we can examine the correlations of the seven features contained in the dataset and the heatmap is used to represent the correlation between the features.
Now we look at the training of the model and visualize how the epoch vs loss of training and testing dataset after 50 epochs. According to the observation, the loss cost achieves a stable least at 50 iterations. Hence the 50 iterations are chosen as the optimal amount. The figure 8 shows the outcome of each model’s comparative testing:
Figure 8. epoch vs loss
Lastly, we will examine the prediction of the model and also check how the system perform in predicting the result to help solve the problem that was formulated above. The test data to see the prediction is from the dataset where will take only 1000 data point from 30,000 to 31,000 to see how the prediction will predict the consumption of power. The graph of prediction will be on the apendix due to space here. The LSTM show an accuracy of 0.9012 which was the highest perfomance in the training test.
CONCLUSION
The research suggests a multivariate forecasting model based on the LSTM deep learning technique that takes into account the time series properties of power consumption data. The experimental outcomes directly demonstrate that the method in this study has significantly improved the prediction accuracy of hourly, daily, and monthly electricity consumption, and the noise reduction processing of the household power consumption data can be used to some extent after training and prediction of sample data. The LSTM is implemented to take account of the importance of previous data, it performs the learning more effectively. With the LSTM being flexible it can be seen as more clearly for datasets with time-varying datasets and higher randomness or hard to detect pattern.
The prediction graph shows that LSTM outcome more successful graph at lower peaks and upper peaks, especially at the lower point. The suggested technique, in contrast to the majority of recent work on multivariant forecasting, employed one LSTM structure to extract channel’s feature separately, while the other are cascade structures. Comparative studies have validated its cutting-edge performance. We might acquire more accurate predicting results by deepening the suggested model since training a deeper model is more doable with ongoing hardware improvement. In the future, would like to build multiple model and compare them together to see which model perform best looking the accuracy they produce as the outcome.
REFERENCES
[1] Kim, T.Y. and Cho, S.B., 2018, November. Predicting the household power consumption using CNN-LSTM hybrid networks. In International Conference on Intelligent Data Engineering and Automated Learning (pp. 481-490). Springer, Cham.
[2] Cho, S.B. and Yu, J.M., 2019. Hierarchical modular Bayesian networks for low-power context-aware smartphone. Neurocomputing, 326, pp.100-109.
[3] Hernandez, L., Baladron, C., Aguiar, J.M., Carro, B., Sanchez-Esguevillas, A.J., Lloret, J. and Massana, J., 2014. A survey on electric power demand forecasting: future trends in smart grids, microgrids and smart buildings. IEEE Communications Surveys & Tutorials, 16(3), pp.1460-1495.
[4] Fumo, N. and Biswas, M.R., 2015. Regression analysis for prediction of residential energy consumption. Renewable and sustainable energy reviews, 47, pp.332-343.
[5] Braun, M.R., Altan, H. and Beck, S.B.M., 2014. Using regression analysis to predict the future energy consumption of a supermarket in the UK. Applied Energy, 130, pp.305-313.
[6] Bogomolov, A., Lepri, B., Larcher, R., Antonelli, F., Pianesi, F. and Pentland, A., 2016. Energy consumption prediction using people dynamics derived from cellular network data. EPJ Data Science, 5, pp.1-15.
[7] Jain, R.K., Smith, K.M., Culligan, P.J. and Taylor, J.E., 2014. Forecasting energy consumption of multi-family residential buildings using support vector regression: Investigating the impact of temporal and spatial monitoring granularity on performance accuracy. Applied Energy, 123, pp.168-178.
[8] Li, Y. and Duan, P., 2019. Research on the innovation of protecting intangible cultural heritage in the" internet plus" era. Procedia Computer Science, 154, pp.20-25.
[9] Hebrail, G. and Berard, A., 2012. Individual household electric power consumption data set. UCI Machine Learning Repository.
[10] Chi, D., 2022. Research on electricity consumption forecasting model based on wavelet transform and multi-layer LSTM model. Energy Reports, 8, pp.220-228.
[11] Duan, J., Hou, Z., Fang, S., Lu, W., Hu, M., Tian, X., Wang, P. and Ma, W., 2022. A novel electricity consumption forecasting model based on kernel extreme learning machine-with generalized maximum correntropy criterion. Energy Reports, 8, pp.10113-10124
