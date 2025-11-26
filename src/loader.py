import os
import base64
from io import BytesIO
from pathlib import Path
from pdf2image import convert_from_path
from tqdm import tqdm

def encode_image_base64(image_input) -> str:
    """
    Encodes a PIL Image or file path to base64 string.
    """
    if isinstance(image_input, (str, Path)):
        return base64.b64encode(Path(image_input).read_bytes()).decode('utf-8')
    
    # Assuming PIL Image
    buffered = BytesIO()
    image_input.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def load_pdf(path: str, dpi: int = 300) -> list:
    """
    Converts a PDF or a folder of PDFs to a list of PIL Images.
    """
    images = []
    if os.path.isfile(path):
        if path.lower().endswith('.pdf'):
            images.extend(convert_from_path(path, dpi=dpi))
    elif os.path.isdir(path):
        for pdf_file in tqdm(os.listdir(path), desc="Converting PDFs"):
            if pdf_file.lower().endswith('.pdf'):
                pdf_path = os.path.join(path, pdf_file)
                images.extend(convert_from_path(pdf_path, dpi=dpi))
    
    print(f"\n✅ PDF Conversion complete! Loaded {len(images)} pages.")
    return images