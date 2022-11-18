# Single Value Agreement of [sadness] for 'song2_annotations_sadness.csv'

import pandas as pd
from nltk.metrics import agreement
inputfile="annotations_full_sadness.csv"
print(inputfile)
merged_df=pd.read_csv(inputfile, header=0, index_col=0)
merged_df.columns=["label_a1","label_a2"]

print(merged_df.shape)

labels_matched_df = merged_df.dropna()

data = []
for idx, row in labels_matched_df.iterrows():
    data.append(("a1", idx, row["label_a1"]))
    data.append(("a2", idx, row["label_a2"]))

atask = agreement.AnnotationTask(data=data)

print("Label: [sadness]")
print("Percentage agreement:", atask.avg_Ao())
print("Cohen's Kappa:", atask.kappa())

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
