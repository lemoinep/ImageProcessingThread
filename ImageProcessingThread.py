# Author(s): Dr. Patrick Lemoine

import os
import sys
import cv2
import pywt
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from skimage.metrics import structural_similarity as compare_ssim
from PIL import Image
from pillow_heif import register_heif_opener
import pillow_avif

register_heif_opener()

_sr = None

def init_superres(model_path='ESPCN_x4.pb'):
    """
    Initialize the super-resolution model from a given model file.

    Args:
        model_path (str): Path to the super-resolution model file.

    Returns:
        The initialized super-resolution model object.
    """
    global _sr
    if _sr is None:
        print(f"[INFO] Loading super-resolution model: {model_path}")
        _sr = cv2.dnn_superres.DnnSuperResImpl_create()
        _sr.readModel(model_path)
        base = os.path.basename(model_path).lower()

        # Determine model name based on filename
        if "espcn" in base:
            model_name = "espcn"
        elif "edsr" in base:
            model_name = "edsr"
        elif "fsrcnn" in base:
            model_name = "fsrcnn"
        elif "lapsrn" in base:
            model_name = "lapsrn"
        else:
            raise ValueError("Unknown model name in the file.")

        # Extract scale factor from filename (e.g. _x4)
        scale_pos = base.find("_x")
        if scale_pos == -1:
            raise ValueError("Upscale factor not found in the filename.")
        scale = int(base[scale_pos+2])

        print(f"[INFO] Model: {model_name}, Scale factor: {scale}")

        _sr.setModel(model_name, scale)
        _sr.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
        _sr.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return _sr

def apply_super_resolution(img, apply=True):
    """
    Apply super-resolution upscaling to the image if enabled.

    Args:
        img (np.array): Input image.
        apply (bool): Whether to apply super-resolution.

    Returns:
        np.array: The upscaled image or original if not applied.
    """
    if not apply:
        return img
    global _sr
    if _sr is None:
        raise RuntimeError("Super-resolution model not initialized. Call init_superres() first.")

    # Convert grayscale or 4-channel image to 3-channel BGR as required by model
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    if img.dtype != np.uint8:
        img = img.astype(np.uint8)

    try:
        return _sr.upsample(img)
    except cv2.error as e:
        print(f"[WARNING] Super-resolution failed: {e}")
        return img  # fallback to original image

def CV_EraseBorder_VERTICAL_NEW(source_color, NbPoints, Color, QUp, QDown, QPositiv, QEntete, QRelease):
    """
    Remove vertical borders from the image by scanning rows from top and bottom
    and cropping where color thresholds are met.

    Args:
        source_color (np.array): Input color image.
        NbPoints (int): Number of sampling points along width.
        Color (tuple): Color threshold tuple.
        QUp (bool): Enable scanning from top.
        QDown (bool): Enable scanning from bottom.
        QPositiv (bool): Threshold condition flag (less than or greater than).
        QEntete (bool): Header adjustment flag.
        QRelease (bool): Release flag (unused).

    Returns:
        np.array: Cropped image with vertical borders removed.
    """
    gray = cv2.cvtColor(source_color, cv2.COLOR_BGR2GRAY)
    source = cv2.merge([gray, gray, gray])
    frameH, frameW = source.shape[:2]
    clim = sum(Color)
    NbPointsFound = min(int(0.90 * NbPoints + 1.0), NbPoints)
    pSU, pSD = 0.40, 0.40

    ju = int(pSU * frameH) if QUp else 0
    if QUp:
        while ju > 0:
            i01 = (np.arange(1, NbPoints+1) / (NbPoints+1) * frameW).astype(int)
            c = source[ju, i01].sum(axis=1)
            NbFound = np.sum(c < clim) if QPositiv else np.sum(c > clim)
            if NbFound > NbPointsFound:
                break
            ju -= 1
        if ju > 0:
            ju += 2

    jd = int(pSD * frameH) if QDown else frameH
    if QDown:
        while jd < frameH - 2:
            i01 = (np.arange(1, NbPoints+1) / (NbPoints+1) * frameW).astype(int)
            c = source[jd, i01].sum(axis=1)
            NbFound = np.sum(c < clim) if QPositiv else np.sum(c > clim)
            if NbFound > NbPointsFound:
                break
            jd += 1
        if jd > frameH - 2:
            jd = frameH - 2

    y1, y2 = ju, jd - ju
    LengthSubPicture = y2 - y1
    LimEntente = 190
    Delta = (LimEntente * frameH) // 3040

    # Adjust delta if cropped area is too small or too large relative to image height
    if frameH / float(LengthSubPicture) > 3.0 or LengthSubPicture / float(frameH) > 0.7:
        Delta = 0

    if QEntete and (y1 - Delta) > 0:
        y1 -= Delta
        y2 += Delta

    if y2 < y1:
        y1, y2 = 0, frameH

    return source_color[y1:y1 + y2, :].copy()

def CV_EraseBorder_HORIZONTAL_NEW(source_color, NbPoints, Color, QLeft, QRight, QPositiv, QEntete, QRelease):
    """
    Remove horizontal borders from the image by scanning columns from left and right
    and cropping where color thresholds are met.

    Args:
        source_color (np.array): Input color image.
        NbPoints (int): Number of sampling points along height.
        Color (tuple): Color threshold tuple.
        QLeft (bool): Enable scanning from left.
        QRight (bool): Enable scanning from right.
        QPositiv (bool): Threshold condition flag (less than or greater than).
        QEntete (bool): Header adjustment flag.
        QRelease (bool): Release flag (unused).

    Returns:
        np.array: Cropped image with horizontal borders removed.
    """
    gray = cv2.cvtColor(source_color, cv2.COLOR_BGR2GRAY)
    source = cv2.merge([gray, gray, gray])
    frameH, frameW = source.shape[:2]
    clim = sum(Color)
    NbPointsFound = min(int(0.90 * NbPoints + 1.0), NbPoints)
    pSL, pSR = 0.40, 0.40

    jl = int(pSL * frameW) if QLeft else 0
    if QLeft:
        while jl > 0:
            j01 = (np.arange(1, NbPoints+1) / (NbPoints+1) * frameH).astype(int)
            c = source[j01, jl].sum(axis=1)
            NbFound = np.sum(c < clim) if QPositiv else np.sum(c > clim)
            if NbFound > NbPointsFound:
                break
            jl -= 1
        if jl > 0:
            jl += 2

    jr = int(pSR * frameW) if QRight else frameW
    if QRight:
        while jr < frameW - 2:
            j01 = (np.arange(1, NbPoints+1) / (NbPoints+1) * frameH).astype(int)
            c = source[j01, jr].sum(axis=1)
            NbFound = np.sum(c < clim) if QPositiv else np.sum(c > clim)
            if NbFound > NbPointsFound:
                break
            jr += 1
        if jr > frameW - 2:
            jr = frameW - 2

    x1, x2 = jl, jr - jl
    LengthSubPicture = x2
    LimEntente = 190
    Delta = (LimEntente * frameW) // 3040

    # Adjust delta if cropped area is too small or too large relative to image width
    if frameW / float(LengthSubPicture) > 3.0 or LengthSubPicture / float(frameW) > 0.7:
        Delta = 0

    if QEntete and (x1 - Delta) > 0:
        x1 -= Delta
        x2 += Delta

    if x2 < x1:
        x1, x2 = 0, frameW

    return source_color[:, x1:x1 + x2].copy()


def CV_Grayscale(img):
    """
    Grayscale
    """
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def CV_RemoveNoise(img):
    """
    Noise removal
    """
    return cv2.medianBlur(img,5)
 
    
def CV_Thresholding(img):
    """
    Thresholding
    """
    return cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


def CV_Dilate(img):
    """
    Dilation
    """
    kernel = np.ones((5,5),np.uint8)
    return cv2.dilate(img, kernel, iterations = 1)
    

def CV_Erode(img):
    """
    Erosion
    """
    kernel = np.ones((5,5),np.uint8)
    return cv2.erode(img, kernel, iterations = 1)


def CV_Opening(img):
    """
    Opening - erosion followed by dilation
    """
    kernel = np.ones((5,5),np.uint8)
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)


def CV_Canny(img):
    """
    Canny edge detection
    """
    return cv2.Canny(img, 100, 200)


def CV_AdjustBrightnessContrast(img,brightness=10,contrast=2.3): 
    """
    Adjust BrightnessContrast
    """
    imgR = cv2.addWeighted(img, contrast, np.zeros(img.shape, img.dtype), 0, brightness) 
    return imgR


def CV_AdaptativeContrast(img,clip=9):
    """
    Contrast adaptatif
    """
    lab=cv2.cvtColor(img,cv2.COLOR_BGR2LAB)
    l,a,b=cv2.split(lab)
    clahe=cv2.createCLAHE(clipLimit=clip,tileGridSize=(8,8))
    merged=cv2.merge((clahe.apply(l),a,b))
    dest=cv2.cvtColor(merged,cv2.COLOR_LAB2BGR)
    return (dest)


def CV_ContourDetection(img):
    """
    Contours detection
    """
    dest=255-cv2.Canny(img,100,100,True)
    return (dest)   


def CV_Match(img1, img2):
    """
    Images matching
    """
    return cv2.matchTemplate(img1, img2, cv2.TM_CCOEFF_NORMED)


def CV_Deskew(img):
    """
    Skew correction  
    """
    coords = np.column_stack(np.where(img > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated


def CV_CLAHE(img, clipLimit=2.0, tileGridSize=(8,8), apply=True):
    """
    Enhance image contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization).

    Args:
        img (np.array): Input BGR image.
        clipLimit (float): Contrast limiting threshold.
        tileGridSize (tuple): Size of grid for histogram equalization.
        apply (bool): Whether to apply CLAHE.

    Returns:
        np.array: Contrast-enhanced image or original if not applied.
    """
    if not apply:
        return img
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)


def CV_EnhanceColor(img, apply=True):
    """
    Increase saturation and brightness in HSV color space to enhance colors.

    Args:
        img (np.array): Input BGR image.
        apply (bool): Whether to apply color enhancement.

    Returns:
        np.array: Color-enhanced image or original if not applied.
    """
    if not apply:
        return img
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * 1.3, 0, 255)  # Saturation boost
    hsv[..., 2] = np.clip(hsv[..., 2] * 1.1, 0, 255)  # Brightness boost
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def CV_DehazingDeFoggyCorrection(img, apply=True):
    """
    Apply bilateral filter and contrast adjustment to reduce haze/fog effect.

    Args:
        img (np.array): Input BGR image.
        apply (bool): Whether to apply dehazing.

    Returns:
        np.array: Dehazed image or original if not applied.
    """
    if not apply:
        return img
    dehazed = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
    alpha, beta = 1.2, 10  # Contrast and brightness parameters
    return cv2.convertScaleAbs(dehazed, alpha=alpha, beta=beta)


def CV_Vibrance2D(img, saturation_scale=1.3, brightness_scale=1.1, apply=True):
    """
    Adjust vibrance by scaling saturation and brightness in HSV space.

    Args:
        img (np.array): Input BGR image.
        saturation_scale (float): Factor to scale saturation.
        brightness_scale (float): Factor to scale brightness.
        apply (bool): Whether to apply vibrance adjustment.

    Returns:
        np.array: Vibrance-adjusted image or original if not applied.
    """
    if not apply:
        return img
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * saturation_scale, 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * brightness_scale, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def CV_Sharpen2D(img, alpha=1.5, gamma=0, op=0, apply=True):
    """
    Sharpen the image using a convolution kernel.

    Args:
        img (np.array): Input image.
        alpha (float): Scaling factor for sharpening.
        gamma (float): Added bias.
        op (int): Unused parameter.
        apply (bool): Whether to apply sharpening.

    Returns:
        np.array: Sharpened image or original if not applied.
    """
    if not apply:
        return img
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(img, -1, kernel)
    return cv2.convertScaleAbs(sharpened, alpha=alpha, beta=gamma)


def CV_SaliencyAddWeighted(img, alpha=0.6, beta=0.4, gamma=0, apply=True):
    """
    Blend the image with its saliency map to enhance salient regions.

    Args:
        img (np.array): Input BGR image.
        alpha (float): Weight for original image.
        beta (float): Weight for saliency map.
        gamma (float): Scalar added to each sum.
        apply (bool): Whether to apply saliency weighting.

    Returns:
        np.array: Image blended with saliency map or original if not applied.
    """
    if not apply:
        return img
    saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
    success, saliencyMap = saliency.computeSaliency(img)
    if not success:
        return img
    saliencyMap = (saliencyMap * 255).astype(np.uint8)
    saliencyMap_color = cv2.cvtColor(saliencyMap, cv2.COLOR_GRAY2BGR)
    return cv2.addWeighted(img, alpha, saliencyMap_color, beta, gamma)


def CV_ApplyFilter(img, filter_type, kernel_size):
    """
    Apply a specified filter to an img.

    Parameters:
    - img: numpy.ndarray, input img (grayscale or color)
    - filter_type: int, filter selector
        1 = Homogeneous blur
        2 = Gaussian blur
        3 = Median blur
        4 = Bilateral filter
        5 = Scharr filter (x-gradient)
        6 = Watershed (placeholder, requires markers)
    - kernel_size: int, kernel size or filter parameter

    Returns:
    - filtered img as numpy.ndarray
    """
    if filter_type == 1:
        # Homogeneous blur
        return cv2.blur(img, (kernel_size, kernel_size))
    elif filter_type == 2:
        # Gaussian blur
        return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    elif filter_type == 3:
        # Median blur
        return cv2.medianBlur(img, kernel_size)
    elif filter_type == 4:
        # Bilateral filter
        return cv2.bilateralFilter(img, kernel_size, kernel_size*2, kernel_size//2)
    elif filter_type == 5:
        # Scharr filter (gradient in x direction)
        if len(img.shape) == 3 and img.shape[2] == 3:
            channels = cv2.split(img)
            scharr_channels = [cv2.Scharr(ch, cv2.CV_8U, 1, 0, scale=4) for ch in channels]
            return cv2.merge(scharr_channels)
        else:
            return cv2.Scharr(img, cv2.CV_8U, 1, 0, scale=4)
    elif filter_type == 6:
        # Watershed placeholder (requires markers)
        # Without markers, just return a copy
        return img.copy()
    else:
        # No filter or unknown type, return original
        return img.copy()


def CV_Sharpen2d(source, alpha, gamma, num_op):
    """
    Sharpen an img with optional pre-filtering.

    Parameters:
    - source: numpy.ndarray, input img
    - alpha: float, weight for the filtered source img
    - gamma: float, scalar added to the weighted sum
    - num_op: int, selects pre-filtering operation
        1 = Gaussian blur with kernel 3
        2 = Homogeneous blur with kernel 9
        else = no pre-filtering

    Returns:
    - sharpened img as numpy.ndarray
    """
    def sharpen_kernel(src):
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
        return cv2.filter2D(src, -1, kernel)

    dst = sharpen_kernel(source)

    if num_op == 1:
        source_filtered = CV_ApplyFilter(source, 2, 3)
    elif num_op == 2:
        source_filtered = CV_ApplyFilter(source, 1, 9)
    else:
        source_filtered = source.copy()

    dst_img = cv2.addWeighted(source_filtered, alpha, dst, 1.0 - alpha, gamma)
    return dst_img

def CV_fastNlMeansDenoisingColored(img, h=3, hColor=3, templateWindowSize=7, searchWindowSize=21):
    """
    Apply Non-local Means Denoising on a colored img.

    Parameters:
    - img: numpy.ndarray, input color img (BGR)
    - h: float, parameter regulating filter strength for luminance component (default 3)
    - hColor: float, same as h but for color components (default 3)
    - templateWindowSize: int, size in pixels of the template patch used to compute weights (default 7)
    - searchWindowSize: int, size in pixels of the window used to compute weighted average (default 21)

    Returns:
    - dst: numpy.ndarray, denoised color img
    """
    dst = cv2.fastNlMeansDenoisingColored(img, None, h, hColor, templateWindowSize, searchWindowSize)
    return dst


def detect_and_draw_contours(img, min_contour_area=100):
    """
    Detects contours in the input img and draws them.

    Parameters:
    - img: numpy.ndarray, input color img (BGR)
    - min_contour_area: int, minimum area of contours to keep (filter small contours)

    Returns:
    - img_with_contours: numpy.ndarray, copy of input img with contours drawn
    - contours: list of contours found (each contour is a numpy array of points)
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply binary thresholding
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours by area
    filtered_contours = [c for c in contours if cv2.contourArea(c) > min_contour_area]

    # Draw contours on a copy of the original img
    img_with_contours = img.copy()
    cv2.drawContours(img_with_contours, filtered_contours, -1, (0, 255, 0), 2)

    return img_with_contours, filtered_contours


def detect_contours_advanced(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    output = image.copy()
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > 500:  # filtrer les petits contours
            approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
            cv2.drawContours(output, [approx], -1, (0, 255, 0), 3)
    
    return output, contours, hierarchy


def compare_images(img1, img2):
    """
    Compare two images and return a similarity score (probability between 0 and 1)
    and recommend which image to keep based on sharpness.

    Parameters:
    - img1, img2: numpy.ndarray, input images (color or grayscale)
      They should have the same dimensions.

    Returns:
    - similarity: float, similarity score between 0 (different) and 1 (identical)
    - keep_img: int, 1 or 2 indicating which image is recommended to keep
    """
    # Convert images to grayscale if needed
    if len(img1.shape) == 3:
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    else:
        gray1 = img1.copy()
    if len(img2.shape) == 3:
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    else:
        gray2 = img2.copy()

    # Compute SSIM between the two images
    similarity, _ = compare_ssim(gray1, gray2, full=True)
    # SSIM is in [-1,1], but usually between 0 and 1 for images

    # Compute sharpness of each image using variance of Laplacian
    def sharpness(img):
        return cv2.Laplacian(img, cv2.CV_64F).var()

    sharp1 = sharpness(gray1)
    sharp2 = sharpness(gray2)

    # Recommend keeping the sharper image
    keep_img = 1 if sharp1 >= sharp2 else 2

    return similarity, 



def CV_FFT(img, apply=True):
    """
    Compute the magnitude spectrum of the 2D Fourier transform of the image.

    Args:
        img (np.array): Input image.
        apply (bool): Whether to apply Fourier transform.

    Returns:
        np.array: Magnitude spectrum image or original if not applied.
    """
    if not apply:
        return img

    if len(img.shape) == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img.copy()

    img_float = np.float32(img_gray)
    dft = cv2.dft(img_float, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]) + 1)
    return magnitude_spectrum


def CV_FFT_full(img, apply=True):
    if not apply:
        return img

    if len(img.shape) == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img.copy()

    img_float = np.float32(img_gray)
    dft = cv2.dft(img_float, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    # img_gray.shape (h, w)
    return dft_shift, img_gray.shape


def CV_iFFT(magnitude_spectrum, original_shape):
    """
    Compute the inverse FFT to reconstruct the image from its Fourier magnitude spectrum.

    Args:
        magnitude_spectrum (np.array): The shifted complex DFT output (2 channels).
        original_shape (tuple): The shape of the original grayscale image.

    Returns:
        np.array: The reconstructed image in spatial domain.
    """
    dft_ishift = np.fft.ifftshift(magnitude_spectrum)
    
    img_back = cv2.idft(dft_ishift)
    
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
    
    cv2.normalize(img_back, img_back, 0, 255, cv2.NORM_MINMAX)
    img_back = np.uint8(img_back)
    
    if img_back.shape != original_shape:
        img_back = cv2.resize(img_back, (original_shape[1], original_shape[0]))
    
    return img_back


def wavelet_transform(img, wavelet='db1', level=1):
    """
    Apply 2D Discrete Wavelet Transform (DWT) on an image.

    Args:
        img (np.array): Input grayscale or color image.
        wavelet (str): Wavelet type (e.g. 'haar', 'db1', 'db3', etc.).
        level (int): Decomposition level.

    Returns:
        tuple: Wavelet coefficients (approximation and details).
    """
    if len(img.shape) == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img.copy()

    img_float = np.float32(img_gray)

    # Perform 2D DWT
    coeffs = pywt.wavedec2(img_float, wavelet=wavelet, level=level)
    return coeffs

def plot_wavelet_coeffs(coeffs):
    """
    Plot the approximation and detail coefficients from 2D DWT.
    """
    cA = coeffs[0]
    cD = coeffs[1:]

    plt.figure(figsize=(12, 3))
    plt.subplot(1, len(cD) + 1, 1)
    plt.imshow(cA, cmap='gray')
    plt.title('Approximation')
    plt.axis('off')

    for i, details in enumerate(cD, 2):
        cH, cV, cD = details
        plt.subplot(1, len(cD) + 1, i)
        plt.imshow(np.abs(cH), cmap='gray')
        plt.title(f'Horizontal Detail Level {i-1}')
        plt.axis('off')

    plt.show()

def inverse_wavelet_transform(coeffs, wavelet='db1'):
    """
    Reconstruct the image from 2D DWT coefficients.

    Args:
        coeffs (tuple): Wavelet coefficients.
        wavelet (str): Wavelet type.

    Returns:
        np.array: Reconstructed image.
    """
    img_reconstructed = pywt.waverec2(coeffs, wavelet=wavelet)
    img_reconstructed = np.clip(img_reconstructed, 0, 255)
    return np.uint8(img_reconstructed)


def CV_Sharpen2D_with_watershed(img, alpha=1.5, gamma=0, apply_sharpen=True, apply_watershed=False, apply=True):
    """
    Apply sharpening and watershed segmentation optionally.

    Args:
        img (np.array): Input image.
        alpha (float): Sharpening scale factor.
        gamma (float): Sharpening bias.
        apply_sharpen (bool): Whether to apply sharpening.
        apply_watershed (bool): Whether to apply watershed segmentation.
        apply (bool): Whether to apply any processing.

    Returns:
        np.array: Processed image.
    """
    if not apply:
        return img

    if not apply_sharpen and not apply_watershed:
        return img

    result = img.copy()

    if apply_sharpen:
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
        sharpened = cv2.filter2D(result, -1, kernel)
        result = cv2.convertScaleAbs(sharpened, alpha=alpha, beta=gamma)

    if apply_watershed:
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY) if len(result.shape) == 3 else result
        ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = np.ones((3,3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        ret, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)
        ret, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0
        markers = cv2.watershed(result, markers)
        result[markers == -1] = [0, 0, 255]  # Mark boundaries in red

    return result


def CV_SegmentationWatershed(img, apply=True):
    """
    Perform watershed segmentation and mark segment boundaries in red.

    Args:
        img (np.array): Input color image.
        apply (bool)

    Returns:
        np.array: Image with watershed boundaries marked.
    """
    if not apply:
        return img
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(img, markers)
    result = img.copy()
    result[markers == -1] = [0, 0, 255]
    return result


def segmentation_kmeans(img, k=3, criteria_eps=0.85, max_iter=100):
    """
    Segment image colors using K-means clustering.

    Args:
        img (np.array): Input BGR image.
        k (int): Number of clusters.
        criteria_eps (float): Convergence epsilon.
        max_iter (int): Maximum iterations.
    Returns:
        segmented_img (np.array): Color-quantized segmented image.
        labels_2d (np.array): Cluster label per pixel.
    """
    pixel_vals = img.reshape((-1, 3))
    pixel_vals = np.float32(pixel_vals)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iter, criteria_eps)
    retval, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    segmented_data = centers[labels.flatten()]
    segmented_img = segmented_data.reshape(img.shape)
    labels_2d = labels.reshape((img.shape[0], img.shape[1]))
    return segmented_img, labels_2d


def find_optimal_k_silhouette(img, k_range=range(2, 11), max_iter=100, epsilon=0.85):
    """
    Find the optimal number of clusters k for K-means using silhouette scores.

    Args:
        img (np.array): Input BGR image.
        k_range (iterable): Range of k values to test.
        max_iter (int): Maximum iterations for K-means.
        epsilon (float): Convergence epsilon.

    Returns:
        optimal_k (int): Optimal number of clusters.
        silhouette_scores (list): Silhouette scores for each k.
    """
    pixel_vals = img.reshape((-1, 3))
    pixel_vals = np.float32(pixel_vals)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iter, epsilon)
    silhouette_scores = []

    for k in k_range:
        _, labels, _ = cv2.kmeans(pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        labels = labels.flatten()
        score = silhouette_score(pixel_vals, labels)
        silhouette_scores.append(score)
        print(f"k={k}, silhouette score={score:.4f}")

    optimal_k = k_range[np.argmax(silhouette_scores)]

    plt.figure()
    plt.plot(list(k_range), silhouette_scores, 'bo-')
    plt.xlabel('Number of clusters k')
    plt.ylabel('Silhouette Score')
    plt.title('Optimal k selection by silhouette score')
    plt.show()

    return optimal_k, silhouette_scores


def CV_OilPaintingEffect(img, size=7, dynRatio=1, apply=True):
    """
    Apply oil painting effect to an image.

    Args:
        img (np.array): Input BGR image.
        size (int): Neighborhood size for effect.
        dynRatio (int): Dynamic ratio, higher values increase effect.
        apply (bool)
    Returns:
        np.array: Image with oil painting effect.
    """
    if not apply:
        return img
    if not hasattr(cv2, 'xphoto') or not hasattr(cv2.xphoto, 'oilPainting'):
        raise ImportError("OpenCV xphoto module or oilPainting function not available. Install opencv-contrib-python.")
    return cv2.xphoto.oilPainting(img, size, dynRatio)


def CV_PointillismEffect(img, dot_radius=5, step=10, apply=True):
    """
    Apply pointillism effect by drawing colored dots on a white canvas.

    Args:
        img (np.array): Input BGR image.
        dot_radius (int): Radius of dots.
        step (int): Step size between dots.
        apply (bool)
    Returns:
        np.array: Image with pointillism effect.
    """
    if not apply:
        return img
    height, width = img.shape[:2]
    canvas = 255 * np.ones_like(img)
    for y in range(0, height, step):
        for x in range(0, width, step):
            color = img[y, x].tolist()
            cv2.circle(canvas, (x, y), dot_radius, color, -1, lineType=cv2.LINE_AA)
    return canvas


def CV_AdvancedPointillism(img, num_colors=20, dot_radius=None, step=None, apply=True):
    """
    Apply advanced pointillism effect by reducing color palette and jittering dot positions.

    Args:
        img (np.array): Input BGR image.
        num_colors (int): Number of colors for palette reduction.
        dot_radius (int): Dot radius; auto-calculated if None.
        step (int): Step between dots; auto-calculated if None.
        apply (bool)
    Returns:
        np.array: Image with advanced pointillism effect.
    """
    if not apply:
        return img
    h, w = img.shape[:2]
    if dot_radius is None:
        dot_radius = max(3, min(h, w) // 100)
    if step is None:
        step = dot_radius * 2

    pixel_vals = img.reshape((-1, 3))
    pixel_vals = np.float32(pixel_vals)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)
    _, labels, centers = cv2.kmeans(pixel_vals, num_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    labels = labels.flatten()
    quantized_img = centers[labels].reshape((h, w, 3))
    canvas = 255 * np.ones_like(img)
    rng = np.random.default_rng()

    for y in range(0, h, step):
        for x in range(0, w, step):
            color = quantized_img[y, x].tolist()
            jitter_x = int(rng.integers(-step//3, step//3))
            jitter_y = int(rng.integers(-step//3, step//3))
            cx = np.clip(x + jitter_x, 0, w-1)
            cy = np.clip(y + jitter_y, 0, h-1)
            cv2.circle(canvas, (cx, cy), dot_radius, color, -1, lineType=cv2.LINE_AA)

    return canvas


def CV_Plastify(image, bilateral_d=15, bilateral_sigma_color=75, bilateral_sigma_space=75, sharpen_strength=1.5, saturation_scale=1.3):
    """
    Applies a plastify effect to the given image.

    Params:
    - image: BGR numpy array image
    - bilateral_d: diameter of the neighborhood for the bilateral filter
    - bilateral_sigma_color: color sigma for the bilateral filter
    - bilateral_sigma_space: space sigma for the bilateral filter
    - sharpen_strength: factor to increase sharpness (1.0 = normal)
    - saturation_scale: factor to scale saturation (1.0 = normal)

    Returns:
    - image modified with plastify effect
    """

    # 1. Bilateral smoothing (smooths textures without degrading edges)
    smooth = cv2.bilateralFilter(image, bilateral_d, bilateral_sigma_color, bilateral_sigma_space)

    # 2. Unsharp masking (sharpness): img_sharp = img + strength * (img - blur(img))
    blur = cv2.GaussianBlur(smooth, (0,0), sigmaX=3)
    unsharp = cv2.addWeighted(smooth, 1.0 + sharpen_strength, blur, -sharpen_strength, 0)

    # 3. Convert to HSV to increase saturation
    hsv = cv2.cvtColor(unsharp, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] *= saturation_scale  # increase S (saturation)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
    plastified = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    return plastified


def CV_ResizeWithAspectRatio(img, width=None, height=None, interpolation=cv2.INTER_AREA, apply=True):
    if not apply:
        return img
    (h, w) = img.shape[:2]
    if width is None and height is None:
        return img
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    return cv2.resize(img, dim, interpolation=interpolation)


def CV_Crop(img, px, py, width, height, apply=True):
    if not apply:
        return img
    (h, w) = img.shape[:2]        
    width = max(0,min(width,w-px))
    height = max(0,min(height,h-py))
    img = img[py:(py+height), px:(px+width)]
    return img
    

def CV_SaveImageToWEBP(img, output_path, quality=80):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    pil_img.save(output_path, 'WEBP', quality=quality)

def CV_SaveImageToAVIF(img, output_path, quality=80):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    pil_img.save(output_path, 'AVIF', quality=quality)

def CV_SaveImageToHEIF(img, output_path, quality=80):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    pil_img.save(output_path, 'HEIF', quality=quality)

def CV_LoadImageAVIF(name):
    img_pil = Image.open(name).convert("RGB")
    img_np = np.array(img_pil)
    img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    return img_cv

def auto_compute_limit1(img, percentile=10):
    """
    Automatically compute a threshold limit based on grayscale percentile.

    Args:
        img (np.array): Input BGR image.
        percentile (int): Percentile to compute threshold.
    Returns:
        int: Threshold value clamped between 10 and 60.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    limit = int(np.percentile(gray, percentile))
    return max(10, min(limit, 60))


def process_image(img_path, output_dir, NbPoints, Limit1, QEntete, QEraseBorderVERTICAL, QEraseBorderHORIZONTAL,
                  px,py,
                  width,height,
                  QEnhanceColor, 
                  QDehazingDeFoggyCorrection, 
                  QColorVibrance, 
                  QSharpen, 
                  QSaliencyAddWeighted, 
                  QCLAHE,
                  QWatershed, 
                  QFFT, 
                  QOilPainting,
                  QPointillism,
                  QAdvancedPointillism,
                  QSegmentationWatershed,
                  apply_sr=False, model_path='ESPCN_x4.pb',
                  save_formats=['jpg'], 
                  quality=80):

    img = cv2.imread(img_path)
    if img is None:
        print(f"Error reading image {img_path}")
        return

    LimitInf1 = (Limit1, Limit1, Limit1)
    LimitUp1 = (255 - Limit1, 255 - Limit1, 255 - Limit1)

    if QEraseBorderVERTICAL:
        img = CV_EraseBorder_VERTICAL_NEW(img, NbPoints, LimitInf1, True, True, True, QEntete, True)
        img = CV_EraseBorder_VERTICAL_NEW(img, NbPoints, LimitUp1, True, True, False, QEntete, False)

    if QEraseBorderHORIZONTAL:
        img = CV_EraseBorder_HORIZONTAL_NEW(img, NbPoints, LimitInf1, True, True, True, QEntete, True)
        img = CV_EraseBorder_HORIZONTAL_NEW(img, NbPoints, LimitUp1, True, True, False, QEntete, False)

    img = CV_EnhanceColor(img, QEnhanceColor)
    img = CV_DehazingDeFoggyCorrection(img, QDehazingDeFoggyCorrection)
    img = CV_Vibrance2D(img, 1.4, 1.0, QColorVibrance)
    img = CV_Sharpen2D(img, 0.75, 0, 0, QSharpen)
    img = CV_SaliencyAddWeighted(img, 0.4, 0.6, 0, QSaliencyAddWeighted)
    img = CV_Sharpen2D_with_watershed(img, 1.5, 0, False, True, QWatershed)
    img = CV_CLAHE(img, apply=QCLAHE)
    img = CV_OilPaintingEffect(img, 5, 1, QOilPainting)
    img = CV_PointillismEffect(img, 7, 10, QPointillism)
    img = CV_AdvancedPointillism(img, 20, None, None,QAdvancedPointillism)
    img = CV_SegmentationWatershed(img, QSegmentationWatershed)
    
    QCrop = ((px != None) and (py !=None) and (width != None) and (height != None))
    
    QResize = False
    if (width != None) :
        QResize = True
    if (height != None) :
        QResize = True 
    if QCrop : 
       QResize = False 
       
    img = CV_ResizeWithAspectRatio(img, width, height, cv2.INTER_AREA,QResize)
    
    img = CV_Crop(img, px, py, width, height, QCrop)
       
    img = apply_super_resolution(img, apply=apply_sr)
    
    img = CV_FFT(img,QFFT)
    
    base_name = os.path.splitext(os.path.basename(img_path))[0]

    if not save_formats or save_formats == ['jpg']:
        output_path = os.path.join(output_dir, base_name + '.jpg')
        cv2.imwrite(output_path, img)
    else:
        if 'webp' in save_formats:
            output_path_webp = os.path.join(output_dir, base_name + '.webp')
            CV_SaveImageToWEBP(img, output_path_webp, quality=quality)

        if 'avif' in save_formats:
            output_path_avif = os.path.join(output_dir, base_name + '.avif')
            CV_SaveImageToAVIF(img, output_path_avif, quality=quality)

        if 'heif' in save_formats:
            output_path_heif = os.path.join(output_dir, base_name + '.heif')
            CV_SaveImageToHEIF(img, output_path_heif, quality=quality)



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Image processing with OpenCV super-resolution DNN')
    parser.add_argument('--Path', type=str, help='Directory containing .jpg images')
    parser.add_argument('--NbPoints', type=int, default=10, help='Number of sampling points')
    parser.add_argument('--QEntete', type=int, default=0, help='Header adjustment flag')
    parser.add_argument('--superres', action='store_true', help='Enable super-resolution')
    parser.add_argument('--model', type=str, default='ESPCN_x4.pb', help='Super-resolution model path')
    parser.add_argument('--workers', type=int, default=4, help='Number of parallel threads')
    parser.add_argument('--Limit1', type=int, default=30, help='Color threshold limit')
    parser.add_argument('--QEraseBorderVERTICAL', type=int, default=1, help='Remove vertical borders flag')
    parser.add_argument('--QEraseBorderHORIZONTAL', type=int, default=1, help='Remove horizontal borders flag')
    parser.add_argument('--QEnhanceColor', type=int, default=0, help='Apply color enhancement flag')
    parser.add_argument('--QDehazingDeFoggyCorrection', type=int, default=0, help='Apply dehazing flag')
    parser.add_argument('--QColorVibrance', type=int, default=0, help='Apply vibrance flag')
    parser.add_argument('--QSharpen', type=int, default=0, help='Apply sharpening flag')
    parser.add_argument('--QSaliencyAddWeighted', type=int, default=0, help='Apply saliency weighting flag')
    parser.add_argument('--QCLAHE', type=int, default=0, help='Apply CLAHE flag')
    parser.add_argument('--QWatershed', type=int, default=0, help='Apply watershed segmentation flag')
    
    parser.add_argument('--QFFT', type=int, default=0, help='Apply FFT flag')    
    parser.add_argument('--QOilPainting', type=int, default=0, help='Apply OilPainting flag')
    parser.add_argument('--QPointillism', type=int, default=0, help='Apply Pointillism flag')
    parser.add_argument('--QAdvancedPointillism', type=int, default=0, help='Apply Advanced Pointillism flag')
    parser.add_argument('--QSegmentationWatershed', type=int, default=0, help='Apply Segmentation Watershed flag')
    
    parser.add_argument('--px', type=int, default=None, help='posx')
    parser.add_argument('--py', type=int, default=None, help='posy')
    parser.add_argument('--width', type=int, default=None, help='new width')
    parser.add_argument('--height', type=int, default=None, help='new height')
    
    parser.add_argument('--save_formats', nargs='+', default=['jpg'], help='Formats to save: jpg, webp, avif, heif')
    parser.add_argument('--quality', type=int, default=80, help='Compression quality (0-100) for formats supporting it')
        

    args = parser.parse_args()

    input_dir = args.Path
    NbPoints = args.NbPoints
    QEntete = args.QEntete
    apply_sr = args.superres
    model_path = args.model
    workers = args.workers
    Limit1 = args.Limit1
    QEraseBorderVERTICAL = args.QEraseBorderVERTICAL
    QEraseBorderHORIZONTAL = args.QEraseBorderHORIZONTAL
    QEnhanceColor = args.QEnhanceColor
    QDehazingDeFoggyCorrection = args.QDehazingDeFoggyCorrection
    QColorVibrance = args.QColorVibrance
    QSharpen = args.QSharpen
    QSaliencyAddWeighted = args.QSaliencyAddWeighted
    QCLAHE = args.QCLAHE
    QWatershed = args.QWatershed
    QFFT = args.QFFT
    
    QOilPainting = args.QOilPainting
    QPointillism = args.QPointillism
    QAdvancedPointillism = args.QAdvancedPointillism    
    QSegmentationWatershed = args.QSegmentationWatershed
    
    width = args.width 
    height = args.height 
    px = args.px
    py = args.py

    if not os.path.isdir(input_dir):
        print(f"Directory {input_dir} does not exist.")
        sys.exit(1)

    output_dir = os.path.join(input_dir, 'DATA')
    os.makedirs(output_dir, exist_ok=True)

    if apply_sr:
        init_superres(model_path)

    files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.lower().endswith('.jpg')]

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(process_image, f, output_dir, 
                                   NbPoints, Limit1, QEntete, QEraseBorderVERTICAL, QEraseBorderHORIZONTAL,
                                   px,py,
                                   width,height,
                                   QEnhanceColor, 
                                   QDehazingDeFoggyCorrection, 
                                   QColorVibrance, 
                                   QSharpen, 
                                   QSaliencyAddWeighted, 
                                   QCLAHE,
                                   QWatershed,
                                   QFFT,
                                   QOilPainting,
                                   QPointillism,
                                   QAdvancedPointillism,
                                   QSegmentationWatershed,
                                   apply_sr, model_path,
                                   save_formats=args.save_formats,
                                   quality=args.quality) 
                   for f in files]

        # Progress bar with tqdm
        for _ in tqdm(as_completed(futures), total=len(futures), desc="Processing images"):
            pass

    print("Processing completed.")
