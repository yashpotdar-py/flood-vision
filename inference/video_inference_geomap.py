import os
import cv2
import torch
import torch.nn.functional as F
import numpy as np
import json
import folium
from tqdm import tqdm
from shapely.geometry import Polygon, mapping
from models.deeplab_model import get_deeplab_model
from datasets.floodnet_dataset_4class import CLASS_MAP

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------
# Config
# ------------------------------
VIDEO_PATH = "data/videos/test2.mp4"
OUTPUT_VIDEO_PATH = "data/outputs/flood_segmented2.mp4"
MASKS_DIR = "data/outputs/masks"
GEOJSON_DIR = "data/outputs/geojson"
os.makedirs(MASKS_DIR, exist_ok=True)
os.makedirs(GEOJSON_DIR, exist_ok=True)

DRONE_GPS = (19.0760, 72.8777)  # example: drone center lat/lon
CAMERA_FOV = 90  # degrees
IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080

# Color map for overlay
CLASS_COLORS = {
    0: (0, 0, 0),
    1: (255, 0, 0),
    2: (255, 255, 0),
    3: (0, 0, 255)
}


# ------------------------------
# Helper Functions
# ------------------------------
def mask_to_overlay(mask, frame, alpha=0.5):
    color_mask = np.zeros_like(frame, dtype=np.uint8)
    for cls, color in CLASS_COLORS.items():
        color_mask[mask == cls] = color
    blended = cv2.addWeighted(frame, 1 - alpha, color_mask, alpha, 0)
    return blended


def preprocess_frame(frame, size=(512, 512)):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, size)
    img = img.astype(np.float32) / 255.0
    img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    return img


def pixel_to_geo(x_pixel, y_pixel, drone_gps, image_width, image_height, fov):
    lat0, lon0 = drone_gps
    # Simple linear approximation; replace with real camera model for angled shots
    lat = lat0 + (y_pixel / image_height) * 0.001
    lon = lon0 + (x_pixel / image_width) * 0.001
    return lat, lon


def mask_to_polygons(mask):
    from skimage import measure
    polygons = []
    contours = measure.find_contours(mask, 0.5)
    for contour in contours:
        poly = [(x, y) for y, x in contour]  # swap axes
        polygons.append(poly)
    return polygons


# ------------------------------
# Main Function
# ------------------------------
def run_inference_and_geomap(video_path, output_path=None, display=False):
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

    geojson_features = []

    for _ in tqdm(range(frame_count), desc="Processing frames"):
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

        # Save mask
        mask_filename = f"frame_{int(cap.get(cv2.CAP_PROP_POS_FRAMES)):04d}.npy"
        np.save(os.path.join(MASKS_DIR, mask_filename), mask)

        # Convert mask to polygons for GeoJSON
        polygons = mask_to_polygons(mask)
        for poly_pixels in polygons:
            poly_coords = [pixel_to_geo(
                x, y, DRONE_GPS, w, h, CAMERA_FOV) for x, y in poly_pixels]
            shapely_poly = Polygon(poly_coords)
            geojson_features.append({
                "type": "Feature",
                "geometry": mapping(shapely_poly),
                "properties": {"frame": int(cap.get(cv2.CAP_PROP_POS_FRAMES))}
            })

        # Overlay for video
        overlay = mask_to_overlay(mask, frame)
        if display:
            try:
                cv2.imshow("Flood Segmentation", overlay)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            except cv2.error:
                display = False
        if out:
            out.write(overlay)

    # Release video
    cap.release()
    if out:
        out.release()
        print(f"✅ Output video saved at: {output_path}")

    if display:
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass

    # ------------------------------
    # Save GeoJSON with loading
    # ------------------------------
    print("💾 Saving GeoJSON...")
    for _ in tqdm(range(1), desc="GeoJSON saving"):
        geojson_path = os.path.join(GEOJSON_DIR, "flooded_areas.geojson")
        with open(geojson_path, "w") as f:
            json.dump({"type": "FeatureCollection",
                      "features": geojson_features}, f)
    print(f"✅ GeoJSON saved to: {geojson_path}")

    # ------------------------------
    # Generate interactive map with progress
    # ------------------------------
    print("🗺️ Creating interactive map...")
    m = folium.Map(location=DRONE_GPS, zoom_start=16)
    for feature in tqdm(geojson_features, desc="Adding polygons to map"):
        folium.Polygon(
            locations=[(lat, lon)
                       for lat, lon in feature["geometry"]["coordinates"][0]],
            color="blue",
            fill=True,
            fill_opacity=0.5
        ).add_to(m)

    map_path = os.path.join(GEOJSON_DIR, "flood_map.html")
    m.save(map_path)
    print(f"✅ Interactive map saved to: {map_path}")


if __name__ == "__main__":
    run_inference_and_geomap(VIDEO_PATH, OUTPUT_VIDEO_PATH, display=False)
