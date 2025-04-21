# MathFi - Handwritten Math Symbol Recognition

> **🚧 WORK IN PROGRESS 🚧**  
> This project is currently under active development. Features may be incomplete or not functioning as expected. 
> We welcome contributions and feedback as we continue to improve the application.

MathFi is an open-source application that recognizes handwritten mathematical symbols and equations using computer vision and deep learning.

## Features (In Development)

- Upload an image of a handwritten math equation
- Image preprocessing to enhance recognition accuracy
- Recognizes digits and common mathematical symbols using a custom CNN model
- Enhanced recognition of advanced mathematical symbols (π, λ, ∂, ∑, √, ∞, ∫, ≤, ≥)
- Visualization of detected symbols with confidence scores
- Streamlit-based user interface for easy interaction

## Installation

1. Clone this repository:
```bash
git clone https://github.com/CodingFreeze/MathFi.git
cd MathFi
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
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

This will launch the Streamlit web interface on port 8501. You can then:

1. Upload an image of a handwritten equation
2. View the recognized symbols
3. See the symbol probabilities and confidence scores

### Command-line options

```bash
python run.py --help
```

Available options:
- `--train`: Train the model before running the app
- `--port`: Specify the port (default: 8501)
- `--no-browser`: Don't open a browser window

## Deployment

### Deploy to Vercel

To deploy this application to Vercel:

1. Fork this repository to your GitHub account
2. Sign up or log in to [Vercel](https://vercel.com/)
3. Click "New Project" and import the repository
4. Select the "Python" framework preset
5. Configure the following settings:
   - Build Command: `pip install -r requirements.txt`
   - Output Directory: `public`
   - Install Command: Leave blank
6. Add the following environment variables:
   - `PYTHONPATH`: `.`
7. Deploy!

> Note: Due to Vercel's timeout limits for serverless functions, the model training functionality is disabled in deployed versions. Only pre-trained models are used.

### Train the recognition models

Train a custom handwritten math symbol recognition model:

```bash
python train_model.py
```

For advanced symbol recognition:

```bash
python train_advanced_model.py
```

For improved model training with enhanced datasets:

```bash
python train_improved_model.py
```

## Project Structure

- `app.py`: Main Streamlit application
- `run.py`: Script to run the application with options
- `train_model.py`: Basic model training script
- `train_cnn_model.py`: Script to train the CNN-based symbol recognition model
- `train_advanced_model.py`: Advanced symbol recognition training
- `enhance_advanced_math.py`: Enhance recognition for advanced math symbols
- `enhance_dataset.py`: Dataset enhancement utilities
- `evaluate_per_class.py`: Evaluate model performance by symbol class
- `models/`: Directory containing the trained models

## Technical Details

### Symbol Recognition

The symbol recognition uses a custom CNN with:
- Input: Grayscale images normalized to a standard size
- Multiple convolutional blocks with batch normalization and dropout
- Dense layers with regularization for stable training
- Output: Multiple classes covering digits and mathematical symbols

### Advanced Symbol Recognition

The project includes specialized training for advanced mathematical symbols:
- Greek letters (π, λ)
- Calculus symbols (∂, ∑, ∫)
- Mathematical operators (√, ∞, ≤, ≥)

### Data Augmentation

During training, various augmentations improve model robustness:
- Random rotation
- Width/height shifts
- Zoom
- Shear transformations

## Datasets

The model is trained on:
- Basic digits and operators
- Advanced mathematical symbols
- Synthetic data generated for better coverage of rare symbols
- Oversampled datasets for balanced training

## Limitations and Future Improvements

Current limitations:
- Works best with clean, well-separated symbols
- Limited to recognized symbol set
- May struggle with complex handwriting styles

Future improvements:
- Extended symbol set including more advanced notation
- Support for continuous equation parsing
- Enhanced segmentation for connected symbols
- Integration with mathematical computation engines

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
