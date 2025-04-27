from PIL import Image


# traitement_crop.py
def crop_image(image_path):
    img = Image.open(image_path)
    cropped = img.crop((10, 10, 100, 100))
    return cropped
