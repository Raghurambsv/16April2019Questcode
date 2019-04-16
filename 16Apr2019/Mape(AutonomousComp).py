from sklearn.ensemble import RandomForestRegressor

# Instantiate model 
rf =  RandomForestRegressor(n_estimators= 5000, random_state=42)

# Train the model on training data
rf.fit(train_features, train_labels);


# # Make Predictions on Test Data

# In[13]:


# Use the forest's predict method on the test data
predictions = rf.predict(test_features)

# Calculate the absolute errors
errors = abs(predictions - test_labels)

# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')


# # Determine Performance Metrics
#######JUST FOR MY TESTING##########
#errors=[5, 8, 12,16, 20]
#errors=np.array(errors)
#test_labels=[40,4,33,23,100]
#test_labels=np.array(test_labels)
#mape = 100 * (errors / test_labels)
#print("before making null to zero",mape)
#mape = np.nan_to_num(mape)
#print("After making null to zero",mape)
# In[14]:


# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / test_labels)
mape = np.nan_to_num(mape)

# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')


 
RandomForestRegressor(n_estimators= 1000, random_state=42)         ===> MAE: 0.18 Accuracy:  99.26 % 

RandomForestRegressor(n_estimators = 100, criterion = 'mse',
  max_depth = None,min_samples_split = 2, min_samples_leaf = 1)    ===> MAE: 0.19 Accuracy:  99.21 %
                      
RandomForestRegressor(n_estimators=10, max_depth = 3,
                      random_state=42)                             ===> MAE: 0.52 Accuracy:  giving DIVIDE BY ERROR    

  

After 2 most imp feature extraction(Not considering all 9 things) ===>MAE: 0.12 Accuracy:  99.46 %         
                      