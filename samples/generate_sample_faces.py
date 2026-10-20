"""
Script to generate sample face test images for demonstration and testing.
"""

import os
from PIL import Image, ImageDraw, ImageFont

def generate_sample_images(output_dir="samples"):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Target Input Face Scan
    img = Image.new("RGB", (400, 400), color=(240, 242, 245))
    draw = ImageDraw.Draw(img)
    
    # Head contour
    draw.ellipse((100, 80, 300, 320), fill=(245, 210, 180), outline=(200, 150, 120), width=3)
    # Eyes
    draw.ellipse((140, 150, 180, 180), fill=(255, 255, 255), outline=(50, 50, 50), width=2)
    draw.ellipse((155, 160, 165, 170), fill=(30, 30, 30))
    draw.ellipse((220, 150, 260, 180), fill=(255, 255, 255), outline=(50, 50, 50), width=2)
    draw.ellipse((235, 160, 245, 170), fill=(30, 30, 30))
    # Nose
    draw.line([(200, 180), (195, 220), (210, 220)], fill=(180, 130, 100), width=3)
    # Mouth / Smile
    draw.arc((150, 230, 250, 270), start=0, end=180, fill=(180, 50, 50), width=4)
    # Hair
    draw.chord((95, 60, 305, 180), start=180, end=360, fill=(50, 35, 25))

    target_path = os.path.join(output_dir, "input_face.jpg")
    img.save(target_path, "JPEG")
    print(f"Sample input face created at: {target_path}")

    # 2. Matching Social Media Candidate Post Image
    img_match = img.resize((400, 400))
    draw_match = ImageDraw.Draw(img_match)
    draw_match.rectangle((10, 10, 390, 40), fill=(29, 161, 242)) # Twitter banner header
    draw_match.text((20, 15), "Twitter/X Verified Post", fill=(255, 255, 255))
    match_path = os.path.join(output_dir, "social_media_match.jpg")
    img_match.save(match_path, "JPEG")
    print(f"Sample matching social post image created at: {match_path}")

if __name__ == "__main__":
    generate_sample_images()
