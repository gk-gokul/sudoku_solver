import cv2
import numpy as np
import matplotlib.pyplot as plt


def load_image_and_extract_digits(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    digits = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if cv2.contourArea(contour) > 50:
            digit_roi = thresh[y:y+h, x:x+w]
            digit_roi_resized = cv2.resize(
                digit_roi, (28, 28), interpolation=cv2.INTER_AREA)
            digits.append(digit_roi_resized)

    return digits


def solve_sudoku(grid):
    solve_sudoku_recursion(grid)


def solve_sudoku_recursion(grid):
    empty_cell = find_empty_cell(grid)
    if not empty_cell:
        return True

    row, col = empty_cell

    for num in range(1, 10):
        if is_valid_move(grid, num, (row, col)):
            grid[row][col] = num

            if solve_sudoku_recursion(grid):
                return True

            grid[row][col] = 0

    return False


def find_empty_cell(grid):
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if grid[i, j] == 0:
                return i, j
    return None


def is_valid_move(grid, num, pos):
    for i in range(len(grid[0])):
        if grid[pos[0]][i] == num and pos[1] != i:
            return False

    for i in range(len(grid)):
        if grid[i][pos[1]] == num and pos[0] != i:
            return False

    subgrid_x = pos[1] // 3
    subgrid_y = pos[0] // 3

    for i in range(subgrid_y * 3, subgrid_y * 3 + 3):
        for j in range(subgrid_x * 3, subgrid_x * 3 + 3):
            if grid[i][j] == num and (i, j) != pos:
                return False

    return True


def display_image(image, title='Image'):
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()


def main(image_path):
    # Load image and extract digits
    digits = load_image_and_extract_digits(image_path)

    if not digits:
        print("No digits found in the image.")
        return

    # Convert extracted digits to a 9x9 grid
    grid = np.zeros((9, 9), dtype=int)
    for idx, digit_image in enumerate(digits):
        row = idx // 9
        col = idx % 9

        # Preprocess the digit image
        digit_processed = preprocess_digit_image(digit_image)

        # Predict the digit
        digit = predict_digit(digit_processed)

        grid[row, col] = digit

    # Solve Sudoku puzzle
    solve_sudoku(grid)

    # Print solved Sudoku grid
    for row in grid:
        print(row)


def preprocess_digit_image(digit_image):
    if len(digit_image.shape) == 3:  # Color image (BGR or RGB)
        # Convert to grayscale
        gray = cv2.cvtColor(digit_image, cv2.COLOR_BGR2GRAY)
    else:  # Grayscale image
        gray = digit_image

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Apply adaptive thresholding
    _, thresholded = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return thresholded


def predict_digit(digit_image):
    # Find contours in the digit image
    contours, _ = cv2.findContours(
        digit_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # If no contour found, return 0 (empty cell)
    if not contours:
        return 0

    # Find the contour with the largest area
    max_contour = max(contours, key=cv2.contourArea)

    # Get bounding box of the contour
    x, y, w, h = cv2.boundingRect(max_contour)

    # Extract the digit region
    digit_roi = digit_image[y:y+h, x:x+w]

    # Resize the digit region to 28x28
    digit_resized = cv2.resize(digit_roi, (28, 28))

    # Perform any additional preprocessing if necessary

    # For now, let's assume the digit is correctly predicted
    return 1  # Placeholder for the predicted digit


if __name__ == "__main__":
    image_path = "q.png"
    main(image_path)
