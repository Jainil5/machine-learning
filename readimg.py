from PIL import Image

img = Image.open("5.png")

data = list(img.getdata())

print(data)