import csv
import pickle

# Replace 'your_file.pkl' with the path to your .pkl file
file_path = "./data/dakota.pkl"

with open("dakota.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["latitude", "longitude", "embedding"])  # Header
    with open(file_path, "rb") as file:
        data = pickle.load(file)
        for row in data:
            (lat, lon), stack, embedding, (b_x, b_y, b_x1, b_y1) = row
            writer.writerow([lat, lon, ";".join(map(str, embedding))])


# Now 'data' contains the content of your .pkl file
