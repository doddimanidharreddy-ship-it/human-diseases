"""
Extract original high-res assets from Minor review 1ppt.pdf
"""

import os
import pymupdf

def extract_assets():
    pdf_path = r"C:\Users\manid\Downloads\Minor review 1ppt.pdf"
    assets_dir = os.path.join(os.path.dirname(__file__), "assets")
    os.makedirs(assets_dir, exist_ok=True)

    doc = pymupdf.open(pdf_path)

    # 1. Vel Tech Logo from Page 1
    page1 = doc[0]
    imgs_p1 = page1.get_images()
    if imgs_p1:
        img_info = doc.extract_image(imgs_p1[0][0])
        with open(os.path.join(assets_dir, "veltech_logo.png"), "wb") as f:
            f.write(img_info["image"])
        print("[OK] Vel Tech Logo extracted successfully.")

    # 2. Architecture Diagram from Page 9
    page9 = doc[8]
    imgs_p9 = page9.get_images()
    if imgs_p9:
        img_info = doc.extract_image(imgs_p9[0][0])
        with open(os.path.join(assets_dir, "system_architecture_diagram.png"), "wb") as f:
            f.write(img_info["image"])
        print("[OK] System Architecture Diagram extracted successfully.")

    # 3. Render page 9 as high-res PNG just in case
    pix = page9.get_pixmap(dpi=200)
    pix.save(os.path.join(assets_dir, "page9_rendered_architecture.png"))
    print("[OK] Page 9 rendered diagram saved.")

if __name__ == "__main__":
    extract_assets()
