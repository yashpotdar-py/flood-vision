import cv2
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from models.deeplab_model import get_deeplab_model
from datasets.floodnet_dataset_4class import CLASS_MAP

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Color map for classes: background, flooded building, flooded road, water
CLASS_COLORS = {
    0: (0, 0, 0),        # background
    1: (255, 0, 0),      # flooded building
    2: (255, 255, 0),    # flooded road
    3: (0, 0, 255)       # water
}


def mask_to_overlay(mask, frame, alpha=0.5):
    """Overlay segmentation mask on top of frame."""
    color_mask = np.zeros_like(frame, dtype=np.uint8)
    for cls, color in CLASS_COLORS.items():
        color_mask[mask == cls] = color
    blended = cv2.addWeighted(frame, 1 - alpha, color_mask, alpha, 0)
    return blended


def preprocess_frame(frame, size=(512, 512)):
    """Resize, normalize, and convert frame to tensor."""
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, size)
    img = img.astype(np.float32) / 255.0
    img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    return img


def run_inference(video_path, output_path=None, display=True):
    model = get_deeplab_model(num_classes=4)
    model.load_state_dict(torch.load(
        "checkpoint_deeplab_4class.pth", map_location=DEVICE))
    model.to(DEVICE).eval()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Cannot open video: {video_path}")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(
        f"🎥 Processing video: {video_path} ({frame_count} frames, {fps} FPS)")

    for _ in tqdm(range(frame_count)):
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess
        inp = preprocess_frame(frame).to(DEVICE).float()

        # Inference
        with torch.no_grad():
            pred = model(inp)
            mask = torch.argmax(F.softmax(pred, dim=1),
                                dim=1).squeeze().cpu().numpy()

        # Resize mask to original frame size
        mask = cv2.resize(mask.astype(np.uint8), (w, h),
                          interpolation=cv2.INTER_NEAREST)

        # Overlay
        overlay = mask_to_overlay(mask, frame)

        if display:
            try:
                cv2.imshow("Flood Segmentation", overlay)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            except cv2.error:
                print(
                    "⚠️ OpenCV display not supported in this environment. Continuing in headless mode.")
                display = False

        if out:
            out.write(overlay)

    cap.release()
    if out:
        out.release()
        print(f"✅ Output saved at: {output_path}")

    # Only try to close windows if display was enabled and supported
    if display:
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass


if __name__ == "__main__":
    run_inference(
        video_path="data\\videos\\test2.mp4",
        output_path="data\\outputs\\flood_segmented2.mp4",
        display=False  # set to True only if OpenCV GUI is supported
    )
