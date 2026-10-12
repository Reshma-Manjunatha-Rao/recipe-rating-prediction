import pandas as pd
# ---- PREPERATION FOR REGRESSION ----

foodcom_combined_data = pd.read_excel("V:\\studium\\Master Bioinformatik\\Data Literacy\\project\\preprocessing_2\\foodcom_combined_data_494949_total_in_minutes_steps_word_length_ingredients_amount_recipie_category_label_encoded.xlsx")
foodcom_combined_data.drop(foodcom_combined_data.filter(regex="Unname"), axis=1, inplace=True)
#%%
foodcom_combined_data_copy = foodcom_combined_data.copy()
#foodcom_combined_data = foodcom_combined_data_copy.copy()
#%%
# only use recipies with at least 5 reviews
foodcom_combined_data = foodcom_combined_data[foodcom_combined_data["number_of_reviews_without_zeros"] >= 5]
#%%
# drop some columns that arent needed for now
foodcom_combined_data.drop("RecipeId", axis=1, inplace=True)
foodcom_combined_data.drop("AuthorId", axis=1, inplace=True)
foodcom_combined_data.drop("ingredients", axis=1, inplace=True)
foodcom_combined_data.drop("tags", axis=1, inplace=True)
foodcom_combined_data.drop("average_rating", axis=1, inplace=True)
foodcom_combined_data.drop("0_ratings", axis=1, inplace=True)
foodcom_combined_data.drop("1_ratings", axis=1, inplace=True)
foodcom_combined_data.drop("2_ratings", axis=1, inplace=True)
foodcom_combined_data.drop("3_ratings", axis=1, inplace=True)
foodcom_combined_data.drop("4_ratings", axis=1, inplace=True)
foodcom_combined_data.drop("5_ratings", axis=1, inplace=True)
foodcom_combined_data.drop("balanced_without_zeros", axis=1, inplace=True)
foodcom_combined_data.drop("balanced", axis=1, inplace=True)
foodcom_combined_data.drop("number_of_reviews", axis=1, inplace=True)
foodcom_combined_data.drop("number_of_reviews_without_zeros", axis=1, inplace=True)
#%%
# Detect and exclude outliers
columns = list(foodcom_combined_data.columns)
columns.remove("average_rating_without_zeros")

for c in columns:
    q_low = foodcom_combined_data[c].quantile(0.01)
    q_hi  = foodcom_combined_data[c].quantile(0.99)

    foodcom_combined_data = foodcom_combined_data[(foodcom_combined_data[c] < q_hi) & (foodcom_combined_data[c] > q_low)]

#%%
# ---- REGRESSION TIME ---- 

# Divide data into two groups and resample the data thats missing for the balance
# We experimented with multiple groups; See resampling.py
between_1_and_3 = foodcom_combined_data[foodcom_combined_data["average_rating_without_zeros"] <= 3]
between_3_and_5 = foodcom_combined_data[foodcom_combined_data["average_rating_without_zeros"] > 3]
resampled_between_1_and_3 = between_1_and_3.sample(5000, replace=True)
sample_between_3_and_5 = between_3_and_5.sample(5000)

regression_data = pd.concat([resampled_between_1_and_3, sample_between_3_and_5])
# bring in right order
regression_data = regression_data[["TotalTime", "RecipeCategory", "Calories", "FatContent", "SaturatedFatContent", "CholesterolContent", "SodiumContent", "CarbohydrateContent", "FiberContent", "SugarContent", "ProteinContent", "steps_word_length", "ingredients_amount", "average_rating_without_zeros"]]
#regression_data.sample(frac=1).to_excel("V:\\studium\\Master Bioinformatik\\Data Literacy\\project\\regression_data.xlsx")


# optional: normalise regression data
#for column in list(regression_data.columns):
#    regression_data[column] = regression_data[column]/regression_data[column].max()


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import itertools

# Run multiple regressions with all different combinations of input data
# and see where there is the lowest MSE

# this will always stay the same
y = regression_data.iloc[: , (len(regression_data.columns)-1)]
regression_columns_index_list = list(range(len(regression_data.columns)-1))


combinations_of_columns_indexes_list = []

for i in range(1,len(regression_data.columns)):
    combinations_of_length_i = list(list(tup) for tup in itertools.combinations(regression_columns_index_list, i))
    combinations_of_columns_indexes_list.append(combinations_of_length_i)

combinations_of_columns_indexes_list = list(itertools.chain.from_iterable(combinations_of_columns_indexes_list))

#%%
from tqdm import tqdm
import numpy as np

df_rows = []

for combination in tqdm(combinations_of_columns_indexes_list):  
    X = regression_data.iloc[: , combination]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
    reg = LinearRegression().fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    
    # Calculate the mean squared error and r2 score
    mse = mean_squared_error(y_test, y_pred) # the lower the better
    r2s = r2_score(y_test, y_pred) # the closer to 1.0 the better

    # Permutation testing
    perm_scores = []
    for i in np.arange(0, 10):
        # Shuffle the labels
        y_shuffle = np.copy(y_test)
        np.random.shuffle(y_shuffle)

        # Calculate the MSE
        mse_perm = mean_squared_error(y_test, y_shuffle)
        # Append the MSE to perm_scores
        perm_scores.append(mse_perm)
    
    mse_to_mean_perm_mse_distance = abs(mse - np.mean(perm_scores)) # The higher the better

    # Encode which columns were used in this regression
    df_entry = []
    for i in range(len(regression_columns_index_list)):
        if i in combination:
            df_entry.append(1)
        else:
            df_entry.append(0)
    
    df_entry.append(mse)
    df_entry.append(r2s)
    df_entry.append(mse_to_mean_perm_mse_distance)
    df_rows.append(df_entry)
      
#%%
columns_of_combinations_df = list(regression_data.columns)[0:len(list(regression_data.columns))-1]
columns_of_combinations_df.append("mse")
columns_of_combinations_df.append("r2s")
columns_of_combinations_df.append("mse_to_mean_perm_mse_distance")
combinations_df = pd.DataFrame(df_rows,columns=columns_of_combinations_df)
#combinations_df.to_excel("V:\\studium\\Master Bioinformatik\\Data Literacy\\project\\preprocessing_2\\all_regression_combinations.xlsx")
sorted_comb = combinations_df.sort_values("mse")

# Determine most significant features
contributions_rows = []

for i in range(len(columns_of_combinations_df)-3):
    entry = []
    entry.append(columns_of_combinations_df[i])
    entry.append(((combinations_df.iloc[:, i]) * combinations_df["mse"]).sum())
    entry.append(((combinations_df.iloc[:, i]) * combinations_df["r2s"]).sum())
    entry.append(((combinations_df.iloc[:, i]) * combinations_df["mse_to_mean_perm_mse_distance"]).sum())
    contributions_rows.append(entry)

contribution_columns = ["Columnname", "mse_contribution", "r2s_contribution", "perm_contribution"]
contributions_df = pd.DataFrame(contributions_rows,columns=contribution_columns)
#%%
# here you can see which features contribute most to a good prediction.
# the ones ranked at the top minimize mse or maximize r2s or perm contribution
# so these features are the best. You could use this table to evaluate 
# the quality of engineered features (like steps_word_length or ingredients_amount) 
# and see if they add anything to your regression
# in comparison to real features (like calories)
print(contributions_df.sort_values("mse_contribution"),"\n")
print(contributions_df.sort_values("r2s_contribution", ascending=False),"\n")
print(contributions_df.sort_values("perm_contribution", ascending=False),"\n")
#%%
# interesting plots
import matplotlib.pyplot as plt
from scipy.stats import norm

counts, bins = np.histogram(combinations_df["mse"])
mu, std = norm.fit(combinations_df["mse"])
x = np.linspace(0.9, 1.05, 100)
p = norm.pdf(x, mu, std)
plt.plot(x, p*115, 'k', linewidth=2, label='Fitted normal distribution')
plt.stairs(counts, bins, fill=True, label='MSE counts')
plt.axvline(np.mean(combinations_df["mse"]), color="red", label='MSE mean')
title = r"MSE of all regressions: $\mu$: {:.2f}, $\sigma$ {:.2f} | min {:.2f}".format(mu, std, min(combinations_df["mse"]))
plt.title(title)
plt.legend(loc=(0.6, 0.79), prop={'size': 8.1})
plt.show()
#%%
# interesting plots
import matplotlib.pyplot as plt
from scipy.stats import norm

counts, bins = np.histogram(combinations_df["r2s"])
mu, std = norm.fit(combinations_df["r2s"])
x = np.linspace(min(combinations_df["r2s"]), max(combinations_df["r2s"]), 100)
p = norm.pdf(x, mu, std)
plt.plot(x, p*90, 'k', linewidth=2, label='Fitted normal distribution')
plt.stairs(counts, bins, fill=True, label='R2S counts')
plt.axvline(np.mean(combinations_df["r2s"]), color="red", label='R2S mean')
title = r"R2 scores: $\mu$: {:.2f}, $\sigma$ {:.2f} | max {:.2f}".format(mu, std, max(combinations_df["r2s"]))
plt.title(title)
plt.legend(loc=(0.6, 0.79), prop={'size': 8.1})
plt.show()

#%%

# ---- NEURAL NETWORK TIME ----
from keras.models import Sequential
from keras.layers import Dense
#from keras.layers import Dropout
import keras_tuner as kt
import numpy as np

def model_builder(hp):
    model = Sequential()
    model.add(Dense(hp.Int("layer_one_param", min_value=16, max_value=512, step=64), input_dim=13, activation='relu'))
    for i in range(hp.Int("num_layers", 1, 10)):
        #hp_activation = hp.Choice("activation_layer"+str(i), values=["relu","tanh","linear",'sigmoid'])
        model.add(Dense(hp.Int("layer_"+str(i)+"_param", min_value=4, max_value=32, step=4), activation="relu"))
        #model.add(Dropout(hp.Float('dropout_chance_' + str(i), min_value=0.0, max_value=0.7, step=0.1)))
    model.add(Dense(1, activation='linear'))

    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

hyperband_tuner = kt.Hyperband(
    model_builder,
    objective='val_loss',
    max_epochs=40,
    hyperband_iterations=5,
    directory='custom_hyperband_mlp',
    project_name='custom_hyperband')

hyperband_tuner.search_space_summary()

#%%
X = regression_data.iloc[: , :len(regression_data.columns)-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
hyperband_tuner.search(X_train, y_train, validation_data=(X_test, y_test))
#%%
best_model = hyperband_tuner.get_best_models()[0]
best_model.summary()
best_model_hist = best_model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test), workers=16)
# Save the model
best_model.save("saved_models/custom_hyperband")