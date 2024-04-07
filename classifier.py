import cv2
import numpy as np
def extract_features(image):
    # Resize the image to a fixed size
    image = cv2.resize(image, (256, 256))
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Extract SIFT features
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return descriptors
# Step 2: Load and extract features from training images
apple_paths = [r"D:\ap\a.jpg", r'D:\ap\b.jpg', r'D:\ap\c.jpg',r'D:\ap\d.jpg',r'D:\ap\e.jpg']
cat_paths = [r"D:\ban\q.jpg",r"D:\ban\w.jpg",r"D:\ban\r.jpg",r"D:\ban\t.jpg", r"D:\ban\e.jpg" ]
X_train = []
y_train = []
for image_path in apple_paths:
    image = cv2.imread(image_path)
    features = extract_features(image)
    X_train.extend(features)
    y_train.extend([1] * len(features))  # Label 1 for apples
for image_path in cat_paths:
    image = cv2.imread(image_path)
    features = extract_features(image)
    X_train.extend(features)
    y_train.extend([0] * len(features))  # Label 0 for cats
# Convert to numpy arrays
X_train = np.array(X_train)
y_train = np.array(y_train)
# Step 3: Train a classifier (K-Nearest Neighbors)
knn = cv2.ml.KNearest_create()
knn.train(X_train, cv2.ml.ROW_SAMPLE, y_train)
# Step 4: Test the classifier
def test_image(image_path):
    image = cv2.imread(image_path)
    features = extract_features(image)
    _, results, _, _ = knn.findNearest(features, k=1)
    predicted_class = int(results[0])
    return "Apple" if predicted_class == 1 else "banana"
# Test new images
test_image_paths = [r'D:\test.jpg',r'D:\testa.jpg',r'D:\testb.jpg']
for image_path in test_image_paths:
    predicted_label = test_image(image_path)
    print(f"The image {image_path} is classified as: {predicted_label}")
    out = cv2.imread(image_path)
    out = cv2.resize(out, (400, 400))
    new_image = cv2.putText(out,f"{predicted_label}",(20, 20),cv2.FONT_HERSHEY_DUPLEX,1.0,(125, 246, 55),1)
    cv2.imshow('Output', new_image)
    cv2.waitKey(0)
