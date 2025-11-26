from surya.foundation import FoundationPredictor
from surya.recognition import RecognitionPredictor
from surya.detection import DetectionPredictor
from layout_detection.src.config import Config

class ContentDetector:
    def __init__(self):
        print(f"Loading Surya models on {Config.DEVICE}...")
        self.foundation_predictor = FoundationPredictor()
        # Note: RecognitionPredictor was initialized in original script but not used for detection logic
        # Keeping it in case you extend functionality later
        self.recognition_predictor = RecognitionPredictor(self.foundation_predictor) 
        self.detection_predictor = DetectionPredictor(device=Config.DEVICE)
        print("Surya models loaded.")

    def detect_boxes(self, image, threshold=Config.CONF_THRESHOLD):
        results = self.detection_predictor([image])
        det_result = results[0]
        
        # Filter by confidence
        valid_bboxes = [b for b in det_result.bboxes if b.confidence >= threshold]
        return valid_bboxes