import kagglehub

# Download latest version
path = kagglehub.dataset_download("sagyamthapa/handwritten-math-symbols")

print("Path to dataset files:", path)