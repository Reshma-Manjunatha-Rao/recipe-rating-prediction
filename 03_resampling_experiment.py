import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from numpy.random import shuffle
import numpy as np

# adjust for you
dir_path = "D:\\DLProject\\"

foodcom_combined_data = pd.read_excel(dir_path+"foodcom_combined_data_494949_total_in_minutes_steps_word_length_ingredients_amount_recipie_category_label_encoded.xlsx")

plt.hist(foodcom_combined_data.average_rating_without_zeros,edgecolor='black',bins=20)
plt.ylabel("Number of recipes")
plt.xlabel("Average rating")
plt.title("Average recipe rating distribution")
#plt.savefig('rating_distr.png', bbox_inches = "tight")
plt.show()

reg_data_orig = pd.read_excel(dir_path+"regression_data.xlsx")

plt.hist(reg_data_orig.average_rating_without_zeros,edgecolor='black')
plt.ylabel("Number of recipes")
plt.xlabel("Average rating")
plt.title("Average recipe rating distribution")
#plt.savefig('rating_distr.png', bbox_inches = "tight")
plt.show()

# n: how many partitions, dsize: total dataset size
def resample(n, dsize, random_state):
    low_bond = 1
    up_bond = 5
    step = (up_bond-low_bond)/n
    bonds = [1+i*step for i in range(1,n)]
    bonds.append(up_bond)
    part_size = int(dsize/n)
   
    #print(bonds)

    replace = False
    first_part = foodcom_combined_data[foodcom_combined_data["average_rating_without_zeros"] <= bonds[0]]
    if (len(first_part) < part_size):
        replace = True
    regression_data = first_part.sample(part_size, replace=replace, random_state = random_state)
    
    for i in range(1,len(bonds)):
        #print(str(bonds[i-1]) + "   " + str(bonds[i]) )
        replace = False
        curr_between = foodcom_combined_data[(foodcom_combined_data["average_rating_without_zeros"] <= bonds[i]) & 
                                        (foodcom_combined_data["average_rating_without_zeros"] > bonds[i-1])]
        if (len(curr_between) < part_size):
            replace = True
        
        curr_between = curr_between.sample(part_size, replace=replace, random_state = random_state)
       
        regression_data = pd.concat([regression_data,curr_between])

    return regression_data[["TotalTime", "RecipeCategory", "Calories", "FatContent", "SaturatedFatContent", "CholesterolContent", "SodiumContent", "CarbohydrateContent", "FiberContent", "SugarContent", "ProteinContent", "steps_word_length", "ingredients_amount", "average_rating_without_zeros"]]

#example
regression_data = resample(5,10000,100)
plt.hist(regression_data.average_rating_without_zeros,edgecolor='black')
plt.ylabel("Number of recipes")
plt.xlabel("Average rating")
plt.title("Average recipe rating distribution")
plt.show()

def regression(regression_data, random_state):
    X = regression_data.iloc[: , :(len(regression_data.columns)-1)]
    y = regression_data.iloc[: , (len(regression_data.columns)-1)]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state = random_state)
    reg = LinearRegression().fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    
    #print(reg.coef_)

    #%%
    
    # Calculate the mean squared error (you can use the sklearn method above for this)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2s = r2_score(y_test, y_pred)
    adj_r2 = 1-(1-r2s)*(len(y_test)-1)/(len(y_test)-len(X)-1) # https://stackoverflow.com/a/51038943/9298528

    #print("MSE: " + str(round(mse,3)), "RMSE: " + str(round(rmse,3)), r2s, adj_r2)
    return mse

random_state = 100
partitions = [i for i in range(2,20)]
mses = []
for i in partitions:
    regression_data = resample(i,10000,random_state)
    mse = regression(regression_data, random_state)
    mses.append(mse)
    
plt.plot(partitions,mses)
# plot original regression data (two partitions at 2.5)
orig_mse = regression(reg_data_orig, random_state)
plt.plot(0, orig_mse, marker="o", markerfacecolor="red")
plt.xticks(partitions)
print(mses)
print("Original MSE: " + str(orig_mse))

