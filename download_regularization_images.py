import requests
from io import BytesIO
import torch
from datasets import load_dataset
import json
from PIL import Image
from tqdm import tqdm
import os
from torchvision import transforms

print("Loading LAION aesthetics 6.5+...")
dataset = load_dataset("laion/gpt4v-dataset")['train']

datalist = []
for i, data in tqdm(enumerate(dataset)):
    datalist.append(data)

# Define a transform to convert PIL image to tensor
transform = transforms.ToTensor()

class SimpleDatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        try:
            response = requests.get(data['link'], timeout=2)
            response.raise_for_status()  # Raise an exception for HTTP errors
            img = Image.open(BytesIO(response.content)).convert("RGB")
            img = transform(img)  # Convert PIL image to tensor
        except Exception as e:
            print(f"Error fetching image from URL {data['link']}: {e}")
            img = None
        return data, img

# Custom collate_fn to remove the list wrapper
def custom_collate_fn(batch):
    # Since batch_size=1, batch will be a list of one tuple: [(data, img)]
    # We return the first element to get (data, img)
    return batch[0]

# Create dataset and dataloader
dataset = SimpleDatasetWrapper(datalist)
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=1,
    collate_fn=custom_collate_fn  # Apply custom collate_fn
)

# Process and save images
counter = 0
output_dir = "regularization_images"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)  # Create the directory if it doesn't exist
    print(f"Created directory: {output_dir}")

with open("regularization_images.jsonl", "a") as f:
    for data, img in tqdm(dataloader, desc="Processing images"):
        if img is None:
            # print(f"Skipping image {counter} due to fetch error.")
            continue
        else:
            print(f"counter value: {counter}..............................................................")

            file_name = f"{output_dir}/{counter}.jpg"
            try:
                # Convert tensor back to PIL image for saving
                img_pil = transforms.ToPILImage()(img.squeeze(0))
                img_pil.save(file_name, quality=95)

    

                # Write metadata to JSONL file
                counter += 1

                f.write(json.dumps(dict(
                    file_name=file_name,
                    caption=data['caption'],
                )) + "\n")
            except Exception as e:
                print(f"Error saving image {file_name}: {e}")