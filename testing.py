import torch
from torchvision import transforms
from PIL import Image
import copy
import torch
from torch import nn
# Load the image
img = Image.open("datasets/veri/image_query/0002_c008_00084510_0.jpg")

# Define the transform pipeline
transform = transforms.Compose([
    transforms.ColorJitter(contrast=0.5),
    transforms.ToTensor(),
])

# Apply the transform to the image
img_transformed = transform(img)

# Convert the transformed tensor back to a PIL image
img_transformed_pil = transforms.ToPILImage()(img_transformed)

# Display the original and transformed images
#img.show()
#img_transformed_pil.show()


# --------------------------------------------- #

tensor = torch.randn(16,3,4,4)
tensor = torch.cat((tensor, torch.zeros(1, 1, 4,4)))
#print(tensor)

avg = nn.AdaptiveAvgPool2d((2, 1))
avg2 = nn.AdaptiveAvgPool2d((1, 1))

#print(avg(tensor))
#print(avg2(tensor))

con = nn.Conv2d(3, 16, 1,  bias=False)