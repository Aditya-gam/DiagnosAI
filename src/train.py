from dataset import prepare_data_loaders, load_data

# Load the data
def get_data_loaders():
    data, all_labels = load_data()
    
    return prepare_data_loaders(data, all_labels)

