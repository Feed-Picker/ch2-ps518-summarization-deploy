import h5py

def print_h5py_group(group, indent=0):
    for key, item in group.items():
        if isinstance(item, h5py.Group):
            print("  " * indent + f"Group: {key}")
            print_h5py_group(item, indent + 1)
        else:
            print("  " * indent + f"Dataset: {key} - Shape: {item.shape}")

filename = 'text_summarization_model.h5'
with h5py.File(filename, 'r') as file:
    print_h5py_group(file)
