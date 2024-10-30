import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("./rice/sinus.png", cv2.IMREAD_GRAYSCALE)
# img = cv2.imread("./rice/noise.png", cv2.IMREAD_GRAYSCALE)
# img = cv2.imread("./rice/whitebl.png", cv2.IMREAD_GRAYSCALE)

median_filtered_img = cv2.medianBlur(img, 3)

plt.imshow(median_filtered_img, cmap='gray')
plt.title('Median filter')
plt.show()

# Fourier transform
dft = cv2.dft(np.float32(median_filtered_img), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)  # Shift zero frequency to the center

# Get the magnitude spectrum for visualization
magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:,:,0], dft_shift[:,:,1]))

# Display the magnitude spectrum
plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Magnitude Spectrum')
plt.show()

# Create a mask with the same size as the image
rows, cols = img.shape
crow, ccol = rows // 2, cols // 2  # center of the image

# Create a mask with all zeros
mask = np.zeros((rows, cols, 2), np.uint8)

# Define the inner and outer radius for the band-pass filter
r_inner = 20  # Inner radius (removes low frequencies)
r_outer = 40  # Outer radius (removes high frequencies)

# Create the band-pass filter
for i in range(rows):
    for j in range(cols):
        distance = np.sqrt((i - crow) ** 2 + (j - ccol) ** 2)
        if r_inner < distance < r_outer:
            mask[i, j] = 1  # Keep frequencies in this range

# Step 3: Apply the mask to the DFT-shifted result
fshift = dft_shift * mask


# Inverse DFT
f_ishift = np.fft.ifftshift(fshift)
img_back = cv2.idft(f_ishift)
img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

# Normalize the result
cv2.normalize(img_back, img_back, 0, 255, cv2.NORM_MINMAX)

# Display the result
plt.imshow(img_back, cmap='gray')
plt.title('Filtered Image')
plt.show()

# _, binary_img = cv2.threshold(img_back, 127, 255, cv2.THRESH_BINARY)

# img_back = np.abs(img_back).astype(np.uint8)

# equalized_img = cv2.equalizeHist(img_back)

# plt.imshow(binary_img, cmap='gray')
# plt.title('Binary Image')
# plt.show()

cv2.destroyAllWindows()
