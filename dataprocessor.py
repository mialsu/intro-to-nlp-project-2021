import pandas as pd
import random

# Load data to memory
anger_train = pd.read_csv('anger_train.tsv', sep = '\t', header = None)
joy_train = pd.read_csv('joy_train.tsv', sep = '\t', header = None)
anger_test = pd.read_csv('anger_test.tsv', sep = '\t', header = None)
joy_test = pd.read_csv('joy_test.tsv', sep = '\t', header = None)

# Concatenate and shuffle data
train = joy_train.append(anger_train, ignore_index = True)
test = joy_test.append(anger_test, ignore_index = True)
data = train.append(test, ignore_index = True)
random.shuffle(data.values)

# Change all not-joy, and not-anger notations to neutral to have three classes in classification
for i in range (len(data)):
    if data.at[i, 0] == "not-joy" or data.at[i,0] == "not-anger":
        data.at[i, 0] = "neutral"

# Save data to .csv-file
data.to_csv('data.csv', index = False)






