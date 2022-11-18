# Percentage Agreement and Kappa Agreement for 'song2_annotations.csv'

import pandas as pd
from nltk.metrics import agreement
inputfile="annotations_full.csv"
print(inputfile)
merged_df=pd.read_csv(inputfile, header=0, index_col=[0,1])
merged_df.columns=["label_a1","label_a2"]

print(merged_df.shape)


# We can drop rows where we don't have two labelers if we want
# . I think Krippendorf can account for missing data though, so this
# . step isn't strictly necessary if you only care about Krippendorf
labels_matched_df = merged_df.dropna()

# Reformat the data into the form AnnotationTask
#  expects.
data = []
for idx, row in labels_matched_df.iterrows():
    data.append(("a1", idx, row["label_a1"]))
    data.append(("a2", idx, row["label_a2"]))

atask = agreement.AnnotationTask(data=data)

print("Percentage agreement:", atask.avg_Ao())
print("Cohen's Kappa:", atask.kappa())
#print("Fleiss's Kappa:", atask.multi_kappa())
#print("Krippendorf's alpha:", atask.alpha())




# This function maps labels into a numeric space,
# . so we can rely on the ordering of labels.
def priority_distance(left_label, right_label):
    mapped_labels = {
        "Critical": 4,
        "High": 3,
        "Medium": 2,
        "Low": 1,
    }
    left_i = mapped_labels[left_label]
    right_i = mapped_labels[right_label]

    return abs(left_i - right_i)
