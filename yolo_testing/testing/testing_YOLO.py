from ultralytics import YOLO
import cv2
# Load a model
model = YOLO(r"C:\Users\ryan2\OneDrive\ryan_vs\Dissertation\Main_system\best.pt")

img = cv2.imread(
    r"C:\Users\ryan2\OneDrive\ryan_vs\Dissertation\yolo_testing\testing\abc.jpg"
)
# Run batched inference on a list of images
results = model([img])  # return a list of Results objects


# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    result.show()  # display to screen
    result.save(filename="abc.jpg")  # save to disk

high_conf_boxes = [box for box in results[0].boxes if box.conf > 0.7]
print(len(high_conf_boxes))
