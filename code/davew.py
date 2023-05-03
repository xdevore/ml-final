from PIL import Image

image = Image.open('/homes/xdevore/ml-final-project/ml-final/data/train/house_specs/2 Funky 2, Kathryn Dion King - Brothers & Sisters - 1993 Radio Edit_0.jpg')
width, height = image.size
print("width:", width, "height:", height)
