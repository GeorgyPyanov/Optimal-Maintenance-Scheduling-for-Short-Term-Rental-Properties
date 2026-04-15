import kagglehub

# Download the latest processed Beijing Airbnb dataset from Kaggle Hub.
# The returned path points to local cached files.
path = kagglehub.dataset_download("zihanli1/beijing-airbnb-dataclean")

print("Path to dataset files:", path)
