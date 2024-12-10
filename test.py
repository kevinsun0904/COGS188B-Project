import json

sb = []

# read file
with open('yelp_dataset/yelp_academic_dataset_business.json') as f:
    for line in f:
        entry = json.loads(line)
        if entry["city"] == "Santa Barbara":
            sb.append(entry)

print(len(sb))

# function to extract location based on long and latitute
locs = [
    (entry["latitude"],entry["longitude"]) for entry in sb
]

# K-means here

# For each cluster, do yada yada
