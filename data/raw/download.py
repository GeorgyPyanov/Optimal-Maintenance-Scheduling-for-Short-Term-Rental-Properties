import kagglehub

# Download latest version
path = kagglehub.dataset_download("zihanli1/beijing-airbnb-dataclean")

print("Path to dataset files:", path)
