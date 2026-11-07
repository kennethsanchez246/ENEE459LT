import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw
import warnings
import os
from scipy.ndimage import convolve, gaussian_filter

MODEL_PATH = "version-RFB-320.onnx"

warnings.filterwarnings("ignore", category=UserWarning)

sess_options = ort.SessionOptions()
sess_options.log_severity_level = 3  

# 3. Use only available providers to prevent the provider fallback warning
desired_providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
available_providers = [p for p in desired_providers if p in ort.get_available_providers()]

# face detection
def hard_nms(box_scores, iou_threshold, top_k=-1, candidate_size=200):
    box_scores = np.asarray(box_scores)
    if box_scores.ndim != 2 or box_scores.shape[1] != 5:
        raise ValueError("box_scores must have shape (N, 5)")
    if candidate_size < 0:
        raise ValueError("candidate_size must be non-negative")

    if box_scores.shape[0] == 0 or candidate_size == 0 or top_k == 0:
        return box_scores[:0]

    scores = box_scores[:, 0]
    boxes = box_scores[:, 1:]

    candidate_boxes = np.argsort(scores)
    candidate_boxes = candidate_boxes[-candidate_size:]
    picked = []

    while len(candidate_boxes) > 0:
        current = candidate_boxes[-1]
        picked.append(current)
        candidate_boxes = candidate_boxes[:-1]

        if (top_k > 0 and len(picked) >= top_k) or len(candidate_boxes) == 0:
            break

        current_box = boxes[current]
        remaining_boxes = boxes[candidate_boxes]

        x1 = np.maximum(current_box[0], remaining_boxes[:, 0])
        y1 = np.maximum(current_box[1], remaining_boxes[:, 1])
        x2 = np.minimum(current_box[2], remaining_boxes[:, 2])
        y2 = np.minimum(current_box[3], remaining_boxes[:, 3])

        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        current_area = max(0, current_box[2] - current_box[0]) * max(0, current_box[3] - current_box[1])
        remaining_area = np.maximum(0, remaining_boxes[:, 2] - remaining_boxes[:, 0]) * np.maximum(0, remaining_boxes[:, 3] - remaining_boxes[:, 1])
        union = current_area + remaining_area - intersection
        iou = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)

        candidate_boxes = candidate_boxes[iou <= iou_threshold]

    return box_scores[picked]


def detect_faces(image_path, output_path="output_detected.png", conf_threshold=0.7):
    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    resized_image = image.resize((320, 240))
    image_array = np.asarray(resized_image).astype(np.float32)
    image_array = (image_array - 127.0) / 128.0
    image_array = np.transpose(image_array, (2, 0, 1))
    image_array = np.expand_dims(image_array, axis=0)

    model_path = MODEL_PATH
    if not os.path.isfile(model_path):
        model_path = os.path.join(os.path.dirname(__file__), MODEL_PATH)

    session = ort.InferenceSession(
        model_path,
        sess_options=sess_options,
        providers=available_providers,
    )
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: image_array})
    output_names = [output.name for output in session.get_outputs()]
    model_outputs = dict(zip(output_names, outputs))
    if "scores" not in model_outputs or "boxes" not in model_outputs:
        raise ValueError("The ONNX model must expose 'scores' and 'boxes' outputs")

    confidences = np.asarray(model_outputs["scores"])
    boxes = np.asarray(model_outputs["boxes"])
    if confidences.ndim == 3 and confidences.shape[0] == 1:
        confidences = confidences[0]
    if boxes.ndim == 3 and boxes.shape[0] == 1:
        boxes = boxes[0]
    if (
        confidences.ndim != 2
        or confidences.shape[1] < 2
        or boxes.ndim != 2
        or boxes.shape[1] != 4
        or confidences.shape[0] != boxes.shape[0]
    ):
        raise ValueError("The ONNX model must return matching (N, >=2) scores and (N, 4) boxes")

    face_scores = confidences[:, 1]
    mask = face_scores >= conf_threshold

    box_scores = np.column_stack((face_scores[mask], boxes[mask]))
    if len(box_scores) > 0:
        box_scores = hard_nms(box_scores, iou_threshold=0.3)

    draw = ImageDraw.Draw(image)
    for face in box_scores:
        score = face[0]
        x1, y1, x2, y2 = face[1:]
        x1 = int(np.clip(x1 * width, 0, width - 1))
        y1 = int(np.clip(y1 * height, 0, height - 1))
        x2 = int(np.clip(x2 * width, 0, width - 1))
        y2 = int(np.clip(y2 * height, 0, height - 1))
        draw.rectangle((x1, y1, x2, y2), outline="red", width=2)
        draw.text((x1, max(0, y1 - 12)), f"{score:.2f}", fill="red")

    image.save(output_path)


# Edge detection
def non_maximum_suppression(magnitude, angle):
    magnitude = np.asarray(magnitude)
    angle = np.asarray(angle)
    if magnitude.ndim != 2 or angle.shape != magnitude.shape:
        raise ValueError("magnitude and angle must be matching 2D arrays")

    height, width = magnitude.shape
    output = np.zeros((height, width), dtype=np.float32)
    angle = np.rad2deg(angle) % 180

    for i in range(1, height - 1):
        for j in range(1, width - 1):
            direction = angle[i, j]

            if (0 <= direction < 22.5) or (157.5 <= direction < 180):
                neighbor1 = magnitude[i, j - 1]
                neighbor2 = magnitude[i, j + 1]
            elif 22.5 <= direction < 67.5:
                neighbor1 = magnitude[i + 1, j - 1]
                neighbor2 = magnitude[i - 1, j + 1]
            elif 67.5 <= direction < 112.5:
                neighbor1 = magnitude[i - 1, j]
                neighbor2 = magnitude[i + 1, j]
            else:
                neighbor1 = magnitude[i - 1, j - 1]
                neighbor2 = magnitude[i + 1, j + 1]

            if magnitude[i, j] >= neighbor1 and magnitude[i, j] >= neighbor2:
                output[i, j] = magnitude[i, j]

    return output


def hysteresis_thresholding(img, low_thresh, high_thresh):
    img = np.asarray(img)
    if img.ndim != 2:
        raise ValueError("img must be a 2D array")
    if low_thresh > high_thresh:
        raise ValueError("low_thresh must not exceed high_thresh")

    strong = 255
    weak = 50
    output = np.zeros(img.shape, dtype=np.uint8)

    output[img >= high_thresh] = strong
    output[(img >= low_thresh) & (img < high_thresh)] = weak

    # Trace all connected weak pixels starting from strong edges.
    height, width = output.shape
    strong_pixels = list(zip(*np.where(output == strong)))

    while len(strong_pixels) > 0:
        i, j = strong_pixels.pop()
        for row in range(max(0, i - 1), min(height, i + 2)):
            for col in range(max(0, j - 1), min(width, j + 2)):
                if output[row, col] == weak:
                    output[row, col] = strong
                    strong_pixels.append((row, col))

    output[output == weak] = 0
    return output


def canny_scipy(image_path, output_path="edges_scipy.png", low=20, high=60):
    if low > high:
        raise ValueError("low must not exceed high")

    image = Image.open(image_path).convert("L")
    image_array = np.asarray(image).astype(np.float32)
    smoothed = gaussian_filter(image_array, sigma=1.4)

    kx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]])
    ky = np.array([[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]])

    gx = convolve(smoothed, kx)
    gy = convolve(smoothed, ky)
    magnitude = np.hypot(gx, gy)
    if magnitude.max() > 0:
        magnitude = magnitude / magnitude.max() * 255
    angle = np.arctan2(gy, gx)

    thinned = non_maximum_suppression(magnitude, angle)
    edges = hysteresis_thresholding(thinned, low, high)
    Image.fromarray(edges.astype(np.uint8)).save(output_path)
    return edges

if __name__ == "__main__":
    image_path = os.path.join(os.path.dirname(__file__), "face.jpg")
    detect_faces(image_path)
    canny_scipy(image_path)