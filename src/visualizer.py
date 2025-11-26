import matplotlib
# Set backend to Agg (non-interactive) to prevent windows from popping up
matplotlib.use('Agg') 

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import io
from PIL import Image
from src.config import Config

def draw_boxes_on_page(image, ordered_bboxes):
    """
    Draws bounding boxes and ordering numbers on a page.
    Returns: A PIL Image object with the visualizations drawn on it.
    """
    # Create a figure with the same aspect ratio as the image
    # dpi=100 is arbitrary here, we resize the figure to match image pixels
    dpi = 100
    h, w = image.height, image.width
    figsize = w / dpi, h / dpi

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Remove whitespace/margins around the plot
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    ax.imshow(image)

    for i, bbox in enumerate(ordered_bboxes):
        x1, y1, x2, y2 = map(int, bbox.bbox)

        # Expand vertically for visual consistency with the cropping logic
        y1 = max(0, y1 - Config.PADDING_Y)
        y2 = min(image.height, y2 + Config.PADDING_Y)

        # Draw bounding box
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor="red", facecolor="none"
        )
        ax.add_patch(rect)
        
        # Add order number label
        ax.text(
            x1, y1 - 5, 
            f"{i+1}", 
            color="yellow", 
            fontsize=12, 
            weight='bold',
            backgroundcolor="black"
        )

    plt.axis("off")
    
    # Save the plot to a binary buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig) # Close figure to free memory
    
    buf.seek(0)
    
    # Return as PIL Image
    visualized_image = Image.open(buf).convert("RGB")
    return visualized_image

def save_visualized_pdf(visualized_pages, output_path="layout_debug.pdf"):
    """
    Compiles a list of PIL images into a single PDF.
    """
    if not visualized_pages:
        print("⚠️ No pages to save.")
        return

    print(f"💾 Saving visualization PDF to: {output_path}")
    
    # Save the first image and append the rest
    visualized_pages[0].save(
        output_path, 
        "PDF", 
        resolution=100.0, 
        save_all=True, 
        append_images=visualized_pages[1:]
    )
    print("✅ PDF saved successfully.")