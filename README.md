# Early Cycle Life Prediction of Lithium-Metal Batteries with the Aid of Machine Learning

Battery cell manufacturing comprises numerous steps requiring co-optimization, making the development process time consuming and expensive. Lithium-metal batteries with ionic liquid electrolytes are a promising next-generation technology for applications demanding high specific energy and safety, but currently suffer from limited cycle stability. Optimizing the manufacturing process can improve performance, and early cycle life prediction can accelerate this process, reducing cost and time. However, correlating early-stage behaviour with long-term stability is challenging. Machine learning can assist in building these correlations, but feature extraction remains a key hurdle. We present here a set of features manually extracted from the first cycle with high correlation with the battery cycle life. These features are then used as inputs to a machine learning model based on linear regression. While our dataset contains batteries that have reached end-of-life through two different mechanisms, the model can predict the cycle life with an error of 15.3%. The error decreases to 9.6% when the cells are first sorted by end-of-life mechanism. This work highlights the importance of the early charge-discharge behaviour of lithium metal batteries and how this data can be used to inform on the battery cycle life with a view to greatly reducing experimental workload.

# Battery Dataset
The battery dataset is in the BatteryData folder. The data includes the charge, discharge and EIS data of 49 batteries. 

# Prediction model
In order to rerun the experiment, run:

```
python predict_EIS.py
```
