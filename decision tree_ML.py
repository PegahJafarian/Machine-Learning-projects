from scipy.io import arff
import pandas as pd
import numpy as np
import random
#from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
thyroid_data = arff.loadarff('ThyroidData.arff')
df = pd.DataFrame(thyroid_data[0])

# Show the first 5 observations of the dataFram
df.head()

#The basis function(polynomial)
#def poly_feats(input_values,degree):
 # result=[]
  #a=input_values[0]
  #b=input_values[1]
  #c=input_values[2]
  #d=input_values[3]
  #for i in range(degree):
   # for j in range(degree):
    #  for k in range(degree):
     #   for l in range(degree):
      #    if i+j+k+l<=degree:
       #     result.append((a**i)*(b**j)*(c**k)*(d**l))
  #for s in range(len(input_values)):
   # result.append(input_values[s]**degree)
  #return result

def split_test_train(data, test_portion):
    
    test_size = round(test_portion * len(data))

    indices = data.index.tolist()
    test_indices = random.sample(population=indices, k=test_size)

    test_df = data.loc[test_indices]
    train_df = data.drop(test_indices)
    
    return train_df, test_df


random.seed(0)
train_df, test_df = split_test_train(df, 0.2)

test_df.head()

def check_purity(data):
    
    class_column = data[:, -1]
    unique_classes = np.unique(class_column)

    if len(unique_classes) == 1:
        return True
    else:
        return False
    
def classify_data(data):
    
    class_column = data[:, -1]
    unique_classes, counts_unique_classes = np.unique(class_column, return_counts=True)

    index = counts_unique_classes.argmax()
    classification = unique_classes[index]
    
    return classification

def get_potential_splits(data):
    
    potential_splits = {}
    _, n_columns = data.shape
    for column_index in range(n_columns - 1):          # excluding the last column which is the class
        values = data[:, column_index]
        unique_values = np.unique(values)
        
        potential_splits[column_index] = unique_values
    
    return potential_splits

def split_data(data, split_column, split_value):
    
    split_column_values = data[:, split_column]

    type_of_feature = FEATURE_TYPES[split_column]
    if type_of_feature == "continuous":
        data_below = data[split_column_values <= split_value]
        data_above = data[split_column_values >  split_value]
    
    # feature is categorical   
    else:
        data_below = data[split_column_values == split_value]
        data_above = data[split_column_values != split_value]
    
    return data_below, data_above

def calculate_classification_error(data):
    
    class_column = data[:, -1]
    _, counts = np.unique(class_column, return_counts=True)
    
    error = 1 - max(probabilities)
    
    return error

def calculate_entropy(data):
    
    class_column = data[:, -1]
    _, counts = np.unique(class_column, return_counts=True)

    probabilities = counts / counts.sum()
    entropy = sum(probabilities * -np.log2(probabilities))
     
    return entropy

def calculate_gini(data):
    
    class_column = data[:, -1]
    _, counts = np.unique(class_column, return_counts=True)
    
    probabilities = counts / counts.sum()
    gini = 1 - sum(np.square(probabilities))
    
    return gini

def calculate_overall_classification_error(data_below, data_above):
    
    n = len(data_below) + len(data_above)
    p_data_below = len(data_below) / n
    p_data_above = len(data_above) / n

    overall_classification_error =  (p_data_below * calculate_classification_error(data_below) 
                                     + p_data_above * calculate_classification_error(data_above))
    
    return overall_classification_error

def calculate_overall_entropy(data_below, data_above):
    
    n = len(data_below) + len(data_above)
    p_data_below = len(data_below) / n
    p_data_above = len(data_above) / n

    overall_entropy =  (p_data_below * calculate_entropy(data_below) 
                      + p_data_above * calculate_entropy(data_above))
    
    return overall_entropy

def calculate_overall_gini(data_below, data_above):
    
    n = len(data_below) + len(data_above)
    p_data_below = len(data_below) / n
    p_data_above = len(data_above) / n

    overall_gini =  (p_data_below * calculate_gini(data_below) 
                      + p_data_above * calculate_gini(data_above))
    
    return overall_gini

def determine_best_split_IG(data, potential_splits):
    
    overall_entropy = 9999
    for column_index in potential_splits:
        for value in potential_splits[column_index]:
            data_below, data_above = split_data(data, split_column=column_index, split_value=value)
            current_overall_entropy = calculate_overall_entropy(data_below, data_above)

            if current_overall_entropy <= overall_entropy:
                overall_entropy = current_overall_entropy
                best_split_column = column_index
                best_split_value = value
    
    return best_split_column, best_split_value

def determine_best_split_classification_error(data, potential_splits):
    
    overall_classification_error = 9999
    for column_index in potential_splits:
        for value in potential_splits[column_index]:
            data_below, data_above = split_data(data, split_column=column_index, split_value=value)
            current_classification_error = calculate_overall_classification_error(data_below, data_above)

            if current_classification_error <= overall_classification_error:
                overall_classification_error = current_classification_error
                best_split_column = column_index
                best_split_value = value
    
    return best_split_column, best_split_value

def determine_best_split_Gini_index(data, potential_splits):
    
    overall_gini = 9999
    for column_index in potential_splits:
        for value in potential_splits[column_index]:
            data_below, data_above = split_data(data, split_column=column_index, split_value=value)
            current_overall_gini = calculate_overall_gini(data_below, data_above)

            if current_overall_gini <= overall_gini:
                overall_gini = current_overall_gini
                best_split_column = column_index
                best_split_value = value
    
    return best_split_column, best_split_value

def determine_type_of_feature(df):
    
    feature_types = []
    n_unique_values_treshold = 15
    for feature in df.columns:
        if feature != "Class":
            unique_values = df[feature].unique()

            if (len(unique_values) <= n_unique_values_treshold):
                feature_types.append("categorical")
            else:
                feature_types.append("continuous")
    
    return feature_types
determine_type_of_feature(df)

def decision_tree_algorithm(df, counter=0, min_samples=2, max_depth=3):
    
    # data preparations
    if counter == 0:
        global COLUMN_HEADERS, FEATURE_TYPES
        COLUMN_HEADERS = df.columns
        FEATURE_TYPES = determine_type_of_feature(df)
        data = df.values
    else:
        data = df           
    
    
    # base cases
    if (check_purity(data)) or (len(data) < min_samples) or (counter == max_depth):
        classification = classify_data(data)
        
        return classification

    
    # recursive part
    else:    
        counter += 1

        # helper functions 
        potential_splits = get_potential_splits(data)
        split_column, split_value = determine_best_split_IG(data, potential_splits)
        data_below, data_above = split_data(data, split_column, split_value)
        
        # check for empty data
        if len(data_below) == 0 or len(data_above) == 0:
            classification = classify_data(data)
            return classification
        
        # determine question
        feature_name = COLUMN_HEADERS[split_column]
        type_of_feature = FEATURE_TYPES[split_column]
        if type_of_feature == "continuous":
            question = "{} <= {}".format(feature_name, split_value)
            
        # feature is categorical
        else:
            question = "{} = {}".format(feature_name, split_value)
        
        # instantiate sub-tree
        sub_tree = {question: []}
        
        # find answers (recursion)
        yes_answer = decision_tree_algorithm(data_below, counter, min_samples, max_depth)
        no_answer = decision_tree_algorithm(data_above, counter, min_samples, max_depth)
        
        # If the answers are the same, then there is no point in asking the qestion.
        # This could happen when the data is classified even though it is not pure
        # yet (min_samples or max_depth base case).
        if yes_answer == no_answer:
            sub_tree = yes_answer
        else:
            sub_tree[question].append(yes_answer)
            sub_tree[question].append(no_answer)
        
        return sub_tree
    
tree = decision_tree_algorithm(train_df, max_depth=100)
print(tree, width=5)
example = test_df.iloc[0]
print(example)

def classify_test(example, tree):
    question = list(tree.keys())[0]
    feature_name, comparison_operator, value = question.split(" ")

    # ask question
    if comparison_operator == "<=":
        if example[feature_name] <= float(value):
            answer = tree[question][0]
        else:
            answer = tree[question][1]
    
    # feature is categorical
    else:
        if str(example[feature_name]) == value:
            answer = tree[question][0]
        else:
            answer = tree[question][1]

    # base case
    if not isinstance(answer, dict):
        return answer
    
    # recursive part
    else:
        residual_tree = answer
        return classify_test(example, residual_tree)
    
def calculate_error(df, tree):

    df["classification"] = df.apply(classify_test, args=(tree,), axis=1)
    df["classification_correct"] = df["classification"] == df["Class"]
    
    accuracy = df["classification_correct"].mean()
    error = 1- accuracy
    return error

error = calculate_error(test_df, tree)
print(error)
accuracy = 1-error
print(accuracy)

#a=precision_recall_fscore_support(test_df, tree, average='micro')
#print(a)
print(classification_report(test_df, tree))
print(confusion_matrix(test_df, tree))