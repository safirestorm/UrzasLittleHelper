from ultralytics import YOLO

def train_model():
    # 1. Load the YOLO11-Pose model
    # We use the pretrained 'nano' weights as a starting point
    model = YOLO("yolo11n-pose.pt")

    # 2. Train the model
    results = model.train(
        data="Mtg2-1/data.yaml", # Path to your Roboflow data.yaml
        epochs=10,              # Start with 100, you can stop early if it plateaus
        imgsz=640,               # Standard YOLO resolution
        batch=16,                # Adjust based on your GPU memory (8, 16, 32)
        device="mps",                # Use 0 for your first GPU, or 'cpu' if no GPU
        project="MTG_Scanner",   # Name of the project folder
        name="card_corner_pose", # Name of this specific training run
        save=True,               # Save checkpoints
        plots=True               # Save training charts (loss, mAP, etc.)
    )

if __name__ == "__main__":
    train_model()