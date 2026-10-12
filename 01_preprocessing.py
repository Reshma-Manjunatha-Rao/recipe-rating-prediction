# load all datasets
import pandas as pd

foodcom_recipes = pd.read_csv("V:\\studium\\Master Bioinformatik\\Data Literacy\\project\\recipes.csv")
foodcom_reviews = pd.read_csv("V:\\studium\\Master Bioinformatik\\Data Literacy\\project\\reviews.csv")
foodcom_search_terms_and_tags = pd.read_csv("V:\\studium\\Master Bioinformatik\\Data Literacy\\project\\recipes_w_search_terms.csv")

foodcom_search_terms_and_tags.rename(columns={'id':'RecipeId'}, inplace=True)
foodcom_combined_data = pd.merge(foodcom_recipes, foodcom_search_terms_and_tags, on='RecipeId')

#%%
# Drop after combination

foodcom_combined_data.drop("Name", axis=1, inplace=True)
foodcom_combined_data.drop("AuthorName", axis=1, inplace=True)
foodcom_combined_data.drop("CookTime", axis=1, inplace=True)
foodcom_combined_data.drop("PrepTime", axis=1, inplace=True)
foodcom_combined_data.drop("Description", axis=1, inplace=True)
foodcom_combined_data.drop("Images", axis=1, inplace=True)
foodcom_combined_data.drop("Keywords", axis=1, inplace=True)
foodcom_combined_data.drop("name", axis=1, inplace=True)
foodcom_combined_data.drop("description", axis=1, inplace=True)
foodcom_combined_data.drop("ingredients_raw_str", axis=1, inplace=True)
foodcom_combined_data.drop("DatePublished", axis=1, inplace=True)
foodcom_combined_data.drop("RecipeIngredientQuantities", axis=1, inplace=True)
foodcom_combined_data.drop("AggregatedRating", axis=1, inplace=True)
foodcom_combined_data.drop("ReviewCount", axis=1, inplace=True)
foodcom_combined_data.drop("RecipeServings", axis=1, inplace=True)
foodcom_combined_data.drop("RecipeYield", axis=1, inplace=True)
foodcom_combined_data.drop("serving_size", axis=1, inplace=True)
foodcom_combined_data.drop("servings", axis=1, inplace=True)
foodcom_combined_data.drop("search_terms", axis=1, inplace=True)
foodcom_combined_data.drop("RecipeInstructions", axis=1, inplace=True)

#%%

# Calculate number_of_reviews and review_average

counter = 0
from tqdm import tqdm
for index, recipie in tqdm(foodcom_combined_data.iterrows()):
    all_reviews_of_this_recipie = foodcom_reviews[foodcom_reviews['RecipeId'] == recipie["RecipeId"]]
    all_reviews_of_this_recipie_without_zeros = all_reviews_of_this_recipie[all_reviews_of_this_recipie["Rating"] != 0]
    
    mean_of_all_ratings = all_reviews_of_this_recipie['Rating'].mean()
    mean_of_all_ratings_without_zeros = all_reviews_of_this_recipie_without_zeros['Rating'].mean()
    
    number_of_reviews = len(all_reviews_of_this_recipie)
    number_of_reviews_without_zeros = len(all_reviews_of_this_recipie_without_zeros)
    
    zero_ratings = len(all_reviews_of_this_recipie[all_reviews_of_this_recipie['Rating'] == 0])
    one_ratings = len(all_reviews_of_this_recipie[all_reviews_of_this_recipie['Rating'] == 1])
    two_ratings = len(all_reviews_of_this_recipie[all_reviews_of_this_recipie['Rating'] == 2])
    three_ratings = len(all_reviews_of_this_recipie[all_reviews_of_this_recipie['Rating'] == 3])
    four_ratings = len(all_reviews_of_this_recipie[all_reviews_of_this_recipie['Rating'] == 4])
    five_ratings = len(all_reviews_of_this_recipie[all_reviews_of_this_recipie['Rating'] == 5])
    
    foodcom_combined_data.loc[index, "average_rating"] = mean_of_all_ratings
    foodcom_combined_data.loc[index, "average_rating_without_zeros"] = mean_of_all_ratings_without_zeros
    foodcom_combined_data.loc[index, "number_of_reviews"] = number_of_reviews
    foodcom_combined_data.loc[index, "number_of_reviews_without_zeros"] = number_of_reviews_without_zeros
    foodcom_combined_data.loc[index, "0_ratings"] = zero_ratings
    foodcom_combined_data.loc[index, "1_ratings"] = one_ratings
    foodcom_combined_data.loc[index, "2_ratings"] = two_ratings
    foodcom_combined_data.loc[index, "3_ratings"] = three_ratings
    foodcom_combined_data.loc[index, "4_ratings"] = four_ratings
    foodcom_combined_data.loc[index, "5_ratings"] = five_ratings
    
    # balanced means the min and the max are at most 25% apart
    max_number_of_ratings_without_zeros = max(one_ratings, two_ratings, three_ratings, four_ratings, five_ratings)
    min_number_of_ratings_without_zeros = min(one_ratings, two_ratings, three_ratings, four_ratings, five_ratings)
    twentyfive_percent_of_ratings_without_zeros = number_of_reviews_without_zeros/4
    if max_number_of_ratings_without_zeros - min_number_of_ratings_without_zeros < twentyfive_percent_of_ratings_without_zeros:
        foodcom_combined_data.loc[index, "balanced_without_zeros"] = "True"
    else:
        foodcom_combined_data.loc[index, "balanced_without_zeros"] = "False"
    
    # balanced means the min and the max are at most 25% apart
    max_number_of_ratings = max(zero_ratings, one_ratings, two_ratings, three_ratings, four_ratings, five_ratings)
    min_number_of_ratings = min(zero_ratings, one_ratings, two_ratings, three_ratings, four_ratings, five_ratings)
    twentyfive_percent_of_ratings = number_of_reviews/4
    if max_number_of_ratings - min_number_of_ratings < twentyfive_percent_of_ratings:
        foodcom_combined_data.loc[index, "balanced"] = "True"
    else:
        foodcom_combined_data.loc[index, "balanced"] = "False"
        
    counter += 1
    if counter % 50000 == 0:
        foodcom_combined_data[counter-50000:counter].to_excel("V:\\studium\\Master Bioinformatik\\Data Literacy\\project\\foodcom_combined_data_{}.xlsx".format(counter), engine='xlsxwriter')  


foodcom_combined_data.to_excel("V:\\studium\\Master Bioinformatik\\Data Literacy\\project\\foodcom_combined_data_{}.xlsx".format(counter), engine='xlsxwriter')  
#%%

# convert total time to minutes
foodcom_combined_data["TotalTime"] = foodcom_combined_data["TotalTime"].apply(lambda x: x[2:])

for index,entry in tqdm(foodcom_combined_data.iterrows()):
    if entry["TotalTime"] == "0S":
        foodcom_combined_data.loc[index, "TotalTime"] = 0
    elif entry["TotalTime"].find("H") == -1:
        foodcom_combined_data.loc[index, "TotalTime"] = int(entry["TotalTime"].split("M")[0])
    elif entry["TotalTime"].find("M") == -1:
        foodcom_combined_data.loc[index, "TotalTime"] = int(entry["TotalTime"].split("H")[0]) * 60
    elif entry["TotalTime"].find("H") != -1:
        hours_in_minutes = int(entry["TotalTime"].split("H")[0]) * 60
        minutes = int(entry["TotalTime"].split("H")[1].split("M")[0])
        foodcom_combined_data.loc[index, "TotalTime"] = hours_in_minutes + minutes

foodcom_combined_data.to_excel("V:\\studium\\Master Bioinformatik\\Data Literacy\\project\\foodcom_combined_data_494949_total_in_minutes.xlsx")

#%%
# steps word length
foodcom_combined_data["steps_word_length"] = foodcom_combined_data["steps"].apply(lambda x: len(x.split(" ")))
foodcom_combined_data.drop("steps", axis=1, inplace=True)
foodcom_combined_data.to_excel("V:\\studium\\Master Bioinformatik\\Data Literacy\\project\\foodcom_combined_data_494949_total_in_minutes_steps_word_length.xlsx")

#%%
# number of ingredients 

for index, row in tqdm(foodcom_combined_data.iterrows(), total=foodcom_combined_data.shape[0]):
    try:
        sp = row["RecipeIngredientParts"].split("\"")
        sp = list(filter(("c(").__ne__, sp))
        sp = list(filter((")").__ne__, sp))
        sp = list(filter((", ").__ne__, sp))
        foodcom_combined_data.loc[index, "RecipeIngredientParts"] = ', '.join(sp)

    except Exception as inst:
        print(inst, " ----------- Dataframeindex ", index)
        continue

foodcom_combined_data["ingredients_amount"] = foodcom_combined_data["RecipeIngredientParts"].apply(lambda x: len(x.split(", ")))
foodcom_combined_data.drop("RecipeIngredientParts", axis=1, inplace=True)
foodcom_combined_data.to_excel("V:\\studium\\Master Bioinformatik\\Data Literacy\\project\\foodcom_combined_data_494949_total_in_minutes_steps_word_length_ingredients_amount.xlsx")

#%%
# label encode recipie category
from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
foodcom_combined_data["RecipeCategory"] = label_encoder.fit_transform(foodcom_combined_data["RecipeCategory"])

foodcom_combined_data.to_excel("V:\\studium\\Master Bioinformatik\\Data Literacy\\project\\foodcom_combined_data_494949_total_in_minutes_steps_word_length_ingredients_amount_recipie_category_label_encoded.xlsx")
