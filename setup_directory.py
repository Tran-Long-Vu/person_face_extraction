import os

# List of directories to create
directories = ['./image_data', './models', './outputs']

def create_directories(dirs):
    for directory in dirs:
        if not os.path.exists(directory):
            try:
                # Create the directory
                os.makedirs(directory)
                print(f"Directory '{directory}' created successfully.")
            except Exception as e:
                print(f"Error creating directory '{directory}': {e}")
        else:
            print(f"Directory '{directory}' already exists. Skipping.")

if __name__ == "__main__":
    create_directories(directories)
