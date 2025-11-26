import argparse
import os
from tqdm import tqdm
from src.config import Config
from src.loader import load_pdf, encode_image_base64
from src.page_classifier import LayoutClassifier
from src.detector import ContentDetector
from src.detect_boxes import sort_single_column, sort_two_column_rtl
from src.preprocessor import preprocess_region
from src.visualizer import draw_boxes_on_page, save_visualized_pdf

def detect_layout(input_path):
    print(f"Loading input from: {input_path}")
    pages = load_pdf(input_path, dpi=Config.PDF_DPI)
    if not pages:
        print("No pages found. Exiting.")
        return

    classifier = LayoutClassifier()
    detector = ContentDetector()
    book_pages_cropped = {}
    preprocessed_pages = {}
    # debug_images = [] 
    layout_type = "single_column"
    
    if len(pages) > 0:
        middle_idx = len(pages) // 2
        print(f"Analyzing layout using page {middle_idx + 1}...")
        middle_img_b64 = encode_image_base64(pages[middle_idx])
        try:
            layout_result = classifier.classify(middle_img_b64)
            layout_type = layout_result.get('page_orientation', 'single_column')
            print(f"📄 Detected Layout: {layout_type}")
        except Exception as e:
            print(f"⚠️ Layout detection failed: {e}. Defaulting to single_column.")

    for idx, image in enumerate(tqdm(pages, desc="Processing Pages")):
        bboxes = detector.detect_boxes(image)
        if layout_type == 'single_column':
            ordered_bboxes = sort_single_column(bboxes)
        else:
            ordered_bboxes = sort_two_column_rtl(bboxes)
            
        # --- Visualization Disabled ---
        # viz_img = draw_boxes_on_page(image, ordered_bboxes)
        # debug_images.append(viz_img)
        # ------------------------------

        cropped_images = []
        for bbox in ordered_bboxes:
            x1, y1, x2, y2 = map(int, bbox.bbox)
            y1 = max(0, y1 - Config.PADDING_Y)
            y2 = min(image.height, y2 + Config.PADDING_Y)
            cropped_region = image.crop((x1, y1, x2, y2))
            cropped_images.append(cropped_region)

        book_pages_cropped[idx] = cropped_images
        preprocessed_pages[idx] = [preprocess_region(img) for img in cropped_images]

    # --- PDF Save Disabled ---
    # base_name = os.path.basename(input_path)
    # file_root, _ = os.path.splitext(base_name)
    # output_filename = f"{file_root}_debug.pdf"
    # save_visualized_pdf(debug_images, output_path=output_filename)
    # -------------------------

    print(f"✅ Processing complete. Processed {len(preprocessed_pages)} pages.")
    
    return preprocessed_pages

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arabic OCR Layout Routing Agent")
    parser.add_argument("--input", type=str, required=True, help="Path to PDF or Image")
    args = parser.parse_args()
    
    bounding_boxes = detect_layout(args.input)
    # print(f"Detected bounding boxes: {bounding_boxes}")
    # print(bounding_boxes.keys())
