import cv2

# Read images in grayscale
image1 = cv2.imread(r'E:\git\yolov11_crazytester\imgs\home_tag.png', cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread(r'E:\git\yolov11_crazytester\imgs\home_tag.png', cv2.IMREAD_GRAYSCALE)
# image2 = cv2.imread(r'E:\git\yolov11_crazytester\imgs\home.png', cv2.IMREAD_GRAYSCALE)

# Compute histograms
hist1 = cv2.calcHist([image1], [0], None, [256], [0, 256])
hist2 = cv2.calcHist([image2], [0], None, [256], [0, 256])

# Normalize histograms
hist1 = cv2.normalize(hist1, hist1).flatten()
hist2 = cv2.normalize(hist2, hist2).flatten()

# Compare histograms
methods = ['CORREL', 'CHISQR', 'INTERSECT', 'BHATTACHARYYA']
for method in methods:
    score = cv2.compareHist(hist1, hist2, getattr(cv2, f'HISTCMP_{method}'))
    print(f"{method} comparison score: {score}")
