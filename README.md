# Sudoku Solver using OpenCV and NumPy

This Sudoku solver is built using OpenCV and NumPy libraries in Python. It extracts digits from a Sudoku puzzle image, solves the puzzle, and prints the solved Sudoku grid.

## Dependencies

- Python 3.x
- OpenCV
- NumPy
- Matplotlib

You can install the required libraries using pip:


```python
pip install opencv-python numpy matplotlib
```


## Usage

1. Clone this repository
   
2. Navigate to the cloned directory:
```python
cd sudoku-solver
```
3. Place your Sudoku puzzle image (in PNG format) inside the directory.

4. Open `main.py` and update the `image_path` variable with the filename of your Sudoku puzzle image.

5. Run the script:
```python
python main.py
```


The solved Sudoku grid will be printed in the console output.

## Code Structure

- `main.py`: Main script to load the image, extract digits, solve the Sudoku puzzle, and display the solved grid.

## Customization

You can customize the solver by implementing a digit recognition model directly in `main.py` within the `predict_digit` function. Currently, it returns a placeholder value (1) for the predicted digit.


