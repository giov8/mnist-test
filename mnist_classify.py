from mnist import MNIST 
from PIL import Image, ImageDraw

# LOAD DATASET
# Using the module nmist, load the data to mndata and get the images and its labels
mndata = MNIST('./data')
images, labels = mndata.load_training()

# FOR TESTING PORPOUSES
# Get the (i+1)th and print on screen
i = 4
image, label = images[i], labels[i]
# printing the image
output = Image.new("L", (28,28)) # L (8-bit pixels, black and white), size 28x28
output.putdata(image)
output.save("output.png")

print(label)
