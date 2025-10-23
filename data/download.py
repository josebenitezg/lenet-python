from datasets import load_dataset

ds = load_dataset("ylecun/mnist")
# Save the dataset to the datasets directory
ds.save_to_disk("datasets/mnist")