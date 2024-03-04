import numpy as np
import pytesseract
import cv2
from skimage.segmentation import clear_border
import imutils
from imutils import paths

# Set the Tesseract command path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

class LicensePlateOCR:
    def __init__(self):
        self.allowed_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        self.psm_mode = 7

    def extract_text_from_roi(self, roi):
        tesseract_config = self._generate_tesseract_config()
        return pytesseract.image_to_string(roi, config=tesseract_config)

    def _generate_tesseract_config(self):
        config_params = {
            'char_whitelist': self.allowed_chars,
            'psm': self.psm_mode
        }
        config_str = "-c tessedit_char_whitelist={char_whitelist} --psm {psm}".format(**config_params)
        return config_str

class PyImageSearchANPR (LicensePlateOCR):
    def __init__(self, minAR=4, maxAR=5):
        super().__init__()
        self.minAR = minAR
        self.maxAR = maxAR
        self.rect_kernel_size = (13, 5)
        self.square_kernel_size = (3, 3)
        self.gaussian_blur_size = (5, 5)
        self.blur_kernel = (5, 5)
        self.morph_kernel_size = (13, 5)
        self.light_morph_kernel = (3, 3)

    def preprocess_image(self, gray):
        # Apply blackhat and compute gradient in one line
        grad_x = self.compute_gradient(self.apply_blackhat(gray, self.rect_kernel_size))

        # Directly process the image for plate detection and enhance plate features
        return self.enhance_plate_features(
            self.process_image_for_plate_detection(grad_x),
            self.apply_light_morphology(gray, self.square_kernel_size)
        )

    def apply_blackhat(self, gray, kernel_size):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        return cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

    def apply_light_morphology(self, gray, kernel_size):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        light = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        return cv2.threshold(light, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    def compute_gradient(self, blackhat):
        grad_x = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
        grad_x = np.absolute(grad_x)
        min_val, max_val = np.min(grad_x), np.max(grad_x)
        grad_x = 255 * ((grad_x - min_val) / (max_val - min_val))
        return grad_x.astype("uint8")

    # tresholding improvements
    def process_image_for_plate_detection(self, gradient_image):
        blurred_image = self.apply_blur(gradient_image)
        morphed_image = self.apply_morphological_transformation(blurred_image)
        binary_image = self.convert_to_binary(morphed_image)
        return binary_image

    def apply_blur(self, image):
        return cv2.GaussianBlur(image, self.blur_kernel, 0)

    def apply_morphological_transformation(self, image):
        structuring_element = cv2.getStructuringElement(cv2.MORPH_RECT, self.morph_kernel_size)
        return cv2.morphologyEx(image, cv2.MORPH_CLOSE, structuring_element)

    def convert_to_binary(self, image):
        _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        return binary_image

    def enhance_plate_features(self, thresholded_image, light_mask):
        eroded_then_dilated = self.erode_and_dilate(thresholded_image)
        combined_with_light = self.combine_with_light_mask(eroded_then_dilated, light_mask)
        final_refinement = self.final_morphological_refinement(combined_with_light)
        return final_refinement

    def erode_and_dilate(self, image):
        eroded = cv2.erode(image, None, iterations=2)
        return cv2.dilate(eroded, None, iterations=2)

    def combine_with_light_mask(self, image, light_mask):
        return cv2.bitwise_and(image, image, mask=light_mask)

    def final_morphological_refinement(self, image):
        dilated = cv2.dilate(image, None, iterations=2)
        return cv2.erode(dilated, None, iterations=1)
    # tresholding end improvments

    def find_contours(self, thresh, keep=5):
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        return sorted(cnts, key=cv2.contourArea, reverse=True)[:keep]

    def extract_plate(self, gray, contour, clearBorder=False):
        (x, y, w, h) = cv2.boundingRect(contour)
        ar = w / float(h)
        if ar >= self.minAR and ar <= self.maxAR:
            licensePlate = gray[y:y + h, x:x + w]
            return clear_border(cv2.threshold(licensePlate, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1])
        return None

def cleanup_text(text):
    return "".join([c if ord(c) < 128 else "" for c in text]).strip()

def main():
    input_dir = "Images"

    anpr = PyImageSearchANPR()
    imagePaths = sorted(list(paths.list_images(input_dir)))

    for imagePath in imagePaths:
        print("Testing " + imagePath)

        image = cv2.imread(imagePath)
        image = imutils.resize(image, width=600)

        if image is None:
            print(f"Warning: Could not read image from {imagePath}")
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh = anpr.preprocess_image(gray)
        contours = anpr.find_contours(thresh)

        for contour in contours:
            roi = anpr.extract_plate(gray, contour)
            if roi is not None:
                lpText = anpr.extract_text_from_roi(roi)
                print(f"License Plate Detected: {cleanup_text(lpText)}")
                break

if __name__ == "__main__":
    main()

    # python .\Main.py
