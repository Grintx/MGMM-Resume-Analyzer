from pdf2image import convert_from_path
import os

def generate_thumbnail(pdf_path, output_dir="thumbnails"):
    os.makedirs(output_dir, exist_ok=True)
    images = convert_from_path(pdf_path, first_page=1, last_page=1)
    thumbnail_path = os.path.join(output_dir, os.path.basename(pdf_path) + ".jpg")
    images[0].save(thumbnail_path, "JPEG")
    return thumbnail_path
