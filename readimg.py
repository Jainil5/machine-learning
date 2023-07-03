from PIL import Image

img = Image.open("D:\Python Projects\mldl\digit-classifier/digits/0.png")

data = list(img.getdata())

print(data)
print(len(data))