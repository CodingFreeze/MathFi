# MathFi - Handwritten Math Solver

MathFi is an open-source application that recognizes and solves handwritten mathematical equations using computer vision and symbolic computation.

![MathFi Demo](app/static/images/mathfi_demo.gif)

## Features (Planned)

- Upload an image of a handwritten math equation or capture it using your webcam
- Image preprocessing to enhance recognition accuracy
- Recognizes digits and common mathematical symbols using a CNN model
- Automatically segments and identifies individual math symbols
- Solves equations and expressions using SymPy
- Displays step-by-step solutions with LaTeX formatting
- Download solutions as PDF
- Visualization of detected symbols with confidence scores

## Installation

1. Clone this repository:
```bash
git clone https://github.com/CodingFreeze/MathFi.git
cd mathfi
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Run the web application

```bash
python run.py
```

This will launch the Streamlit web interface. You can then:

1. Upload an image of a handwritten equation
2. Or use your webcam to capture an equation
3. View the recognized equation
4. See the step-by-step solution
5. Download the solution as PDF

### Command-line options

```bash
python run.py --help
```

Available options:
- `--train`: Train the model before running the app
- `--port`: Specify the port (default: 8501)
- `--no-browser`: Don't open a browser window

### Train the recognition model

Train a custom handwritten math symbol recognition model:

```bash
python train_cnn_model.py --model-type custom_cnn --epochs 20 --batch-size 32
```

Available model types:
- `custom_cnn`: Standard CNN architecture (default)
- `mobilenet`: MobileNetV2-based architecture for higher accuracy
- `resnet`: ResNet50-based architecture for even higher accuracy

The model will be saved to `models/symbol_recognition_model` and will be automatically used by the application.

## Advanced Usage

### Advanced options in the UI

- **Adaptive Thresholding**: Toggle between Otsu's and adaptive thresholding for different lighting conditions
- **Minimum Symbol Area**: Adjust the minimum area to filter out small noise or detect smaller symbols
- **Symbol Detection Visualization**: See the bounding boxes and recognized symbols overlaid on the image
- **Merge Touching Symbols**: Automatically merge touching or closely positioned symbols
- **Recognition Model**: Select which model type to use for recognition

### Manual Equation Correction

If the automatic recognition isn't perfect, you can manually correct the equation before solving:
1. Upload an image or take a photo
2. View the recognized equation
3. Edit the equation in the text input box if needed
4. The solution will be based on your corrected equation

## Deployment

### Deploy on Streamlit Cloud

1. Push your code to GitHub
2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
3. Connect your GitHub repository
4. Set the main file to `app.py`
5. Add any required secrets (if needed)

### Deploy with Docker

1. Build the Docker image:
```bash
docker build -t mathfi .
```

2. Run the container:
```bash
docker run -p 8501:8501 mathfi
```

3. Access the application at http://localhost:8501

## Project Structure

- `app.py`: Main Streamlit application
- `run.py`: Script to run the application with options
- `train_model.py`: Script to train the model
- `train_cnn_model.py`: Script to train the CNN-based symbol recognition model
- `app/utils/`: Utility functions
  - `image_processing.py`: Image preprocessing and symbol segmentation
  - `recognition.py`: Symbol recognition
  - `equation_solver.py`: Equation parsing and solving
  - `data_preprocessing.py`: Dataset preparation functions
- `app/models/`: Model architecture and training
  - `cnn_model.py`: CNN model architecture
- `tests/`: Test suite

## Technical Details

### Image Processing Pipeline

1. Convert to grayscale
2. Apply Gaussian blur for noise reduction
3. Apply adaptive thresholding for binarization
4. Deskew the image to correct rotation
5. Use morphological operations to clean up the image
6. Find contours to identify individual symbols
7. Merge touching symbols if needed
8. Extract, resize, and normalize symbol images
9. Clean and center each symbol for better recognition

### Symbol Recognition

The symbol recognition uses a CNN with the following architecture:
- Input: 28×28 grayscale images
- 3 convolutional blocks with batch normalization and dropout
- Dense layers with batch normalization and dropout for regularization
- Output: 20 classes (digits 0-9 and mathematical symbols)

Alternative architectures:
- **MobileNetV2**: Pre-trained on ImageNet, adapted for symbol recognition
- **ResNet50**: Pre-trained on ImageNet, adapted for symbol recognition

### Data Augmentation

During training, we apply various augmentations to improve model robustness:
- Random rotation (±10°)
- Width/height shifts
- Zoom
- Shear transformations

### Equation Solving

The equation solving uses SymPy:
1. Parse the recognized expression
2. For equations (containing =):
   - Solve for the unknown variable(s)
   - Generate step-by-step solution
3. For expressions:
   - Simplify the expression
   - Show the computation

## Datasets

The model can be trained on:
- MNIST dataset for digits
- Synthetic math symbols generated with various fonts
- CROHME dataset (if available)
- Kaggle's Handwritten Mathematical Expressions dataset

## Limitations

This is a prototype with some limitations:

- Works best with clean, well-separated symbols
- Currently supports single-line equations
- Limited to basic mathematical operators and variables
- May struggle with complex handwriting styles

## Future Improvements

- Train a more robust symbol recognition model using the full CROHME dataset
- Improve symbol segmentation for overlapping characters
- Add support for more complex mathematical notation (fractions, exponents, etc.)
- Implement multi-line equation support
- Add mobile support with a responsive UI
- Create a browser extension

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

MIT

## Acknowledgments

- [MNIST](http://yann.lecun.com/exdb/mnist/) for the digit dataset
- [CROHME](https://www.isical.ac.in/~crohme/) for math symbol datasets
- [Kaggle Handwritten Math Expressions](https://www.kaggle.com/datasets/xainano/handwritten-mathematical-expressions)
- [SymPy](https://www.sympy.org/) for symbolic mathematics
- [Streamlit](https://streamlit.io/) for the web interface
- [TensorFlow](https://www.tensorflow.org/) for deep learning capabilities
- [OpenCV](https://opencv.org/) for image processing 
