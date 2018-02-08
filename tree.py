import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier,export_graphviz

df=pd.read_csv("dataset.csv")                               #KDDCUP1999 dataset(Pandas library is used for reading the .csv file)

def encode_target(df, target_column):                       #the target is encoded into int because it has a string data type
    df_mod = df.copy()
    targets = df_mod[target_column].unique()                      #List of targets
    map_to_int = {name: n for n, name in enumerate(targets)}      #Assign each target an integer
    df_mod["Target"] = df_mod[target_column].replace(map_to_int)  #Make a new column named Target which contains the integer values of the targets
    return (df_mod, targets)

def encode_feature1(df, feature_column):                        #all the features whose value is in string data type are encoded into int
    df_mod = df.copy()
    features = df_mod[feature_column].unique()
    map_to_int = {name: n for n, name in enumerate(features)}
    df_mod["protocol_type_int"] = df_mod[feature_column].replace(map_to_int)
    return (df_mod,features)

def encode_feature2(df, feature_column):
    df_mod = df.copy()
    features = df_mod[feature_column].unique()
    map_to_int = {name: n for n, name in enumerate(features)}
    df_mod["service_int"] = df_mod[feature_column].replace(map_to_int)
    return (df_mod,features)

def encode_feature3(df, feature_column):
    df_mod = df.copy()
    features = df_mod[feature_column].unique()
    map_to_int = {name: n for n, name in enumerate(features)}
    df_mod["flag_int"] = df_mod[feature_column].replace(map_to_int)
    return (df_mod,features)

def visualize_tree(tree, feature_names):                          #generating the decision tree
    with open("dt.dot", 'w') as f:
        export_graphviz(tree, out_file=f,
                        feature_names=feature_names)

    command = ["dot", "-Tpng", "dt.dot", "-o", "dt.png"]
    try:
        subprocess.check_call(command)
    except:
        exit()


def get_code(tree, feature_names, target_names,                 #generating the code
             spacer_base="    "):
    left      = tree.tree_.children_left
    right     = tree.tree_.children_right
    threshold = tree.tree_.threshold
    features  = [feature_names[i] for i in tree.tree_.feature]
    value = tree.tree_.value

    def recurse(left, right, threshold, features, node, depth):
        spacer = spacer_base * depth
        if (threshold[node] != -2):
            print(spacer + "if ( " + features[node] + " <= " + \
                  str(threshold[node]) + " ) {")
            if left[node] != -1:
                    recurse(left, right, threshold, features,
                            left[node], depth+1)
            print(spacer + "}\n" + spacer +"else {")
            if right[node] != -1:
                    recurse(left, right, threshold, features,
                            right[node], depth+1)
            print(spacer + "}")
        else:
            target = value[node]
            for i, v in zip(np.nonzero(target)[1],
                            target[np.nonzero(target)]):
                target_name = target_names[i]
                target_count = int(v)
                print(spacer + "return " + str(target_name) + \
                      " ( " + str(target_count) + " examples )")

    recurse(left, right, threshold, features, 0, 0)

df2,features1 = encode_feature1(df,"protocol_type")               
df3,features2 = encode_feature2(df2,"service")
df4,features3 = encode_feature3(df3,"flag")
df5,targets = encode_target(df4,"result")

del df5["protocol_type"]                        #delete all the columns having values as string
del df5["service"]
del df5["flag"]
del df5["result"]

features = list(df5.columns[:41])              #print all the features of the dataset

test_data2=[0,1032,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,511,511,0,0,0,0,1,0,0,255,255,1,0,1,0,0,0,0,0,2,9,0]     #data for which the result is to be predicted

y=df5["Target"]
x=df5[features]
dt=DecisionTreeClassifier()
dt.fit(x,y)                         #fit(x,y) function is used to train the features

print ("The result is ",dt.predict(test_data2))     #predicting the result

get_code(dt, features, targets)
visualize_tree(dt,features)
