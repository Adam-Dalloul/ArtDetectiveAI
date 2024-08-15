<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body>

<h1 align="center"><b><u>ArtDetectiveAI</u></b></h1>

<p align="center">
  <img src="https://github.com/user-attachments/assets/3f5440c0-247d-49e9-9f34-6d6494668dec" width="300" height="300" alt="thumbnail">
</p>

<p align="center"><b>ArtDetectiveAI</b> specializes in detecting whether an image (specifically an artistic image) is AI generated or human generated.</p>

<p>With the recent advancements of AI art generation and AI image generation, it is becoming increasingly difficult for humans to differentiate between what is real (human generated) and what is fake (AI generated). This model serves as a way for humans to detect whether images are AI generated or not to prevent plagiarism, dishonest work, and more.</p>

<h2>The Algorithm</h2>

<p>The full dataset can be viewed <a href="https://www.kaggle.com/datasets/kausthubkannan/ai-and-human-art-classification" target="_blank">here</a>. The dataset was slightly modified by splitting some data between the AI and Non_AI training folders to use them for validation instead.</p>

<p>The model utilizes the dataset to classify images into two categories:</p>
<ol>
    <li><b>AI</b></li>
    <li><b>Non_AI</b></li>
</ol>
<p><b>AI</b> refers to images detected as AI art, while <b>Non_AI</b> refers to images generated by humans.</p>

<p>IMAGES OF MODEL DETECTING AI IMAGES AND HUMAN AND ACCURACY</p>

<h2>How It Works</h2>

<p><b>ArtDetectiveAI</b> uses a neural network to classify images. The process involves:</p>
<ol>
    <li><b>Input Image</b>: Upload an image file for classification.</li>
    <li><b>Processing</b>: The image is processed using a pre-trained model (ResNet-18).</li>
    <li><b>Classification</b>: The model predicts the category of the image and provides a confidence score.</li>
</ol>

<h2>Installation</h2>

<p>To get started with <b>ArtDetectiveAI</b>, follow these steps:</p>

<ol>
    <li><b>Clone the Repository</b>:
        <pre><code>git clone https://github.com/YourUsername/ArtDetectiveAI</code></pre>
    </li>
    <li><b>Navigate to the Project Directory</b>:
        <pre><code>cd ArtDetectiveAI</code></pre>
    </li>
    <li><b>Run the Classification Script</b>:
        <pre><code>python3 classify_image.py path/to/image.jpg</code></pre>
    </li>
</ol>

<h2>Script Explanation</h2>

<p>Here’s a breakdown of the <code>classify_image.py</code> script:</p>

<pre><code>
#!/usr/bin/python3
import jetson_inference
import jetson_utils
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("filename", type=str, help="Filename of the image to process")
parser.add_argument("--network", type=str, default="googlenet", help="Model to use (e.g., googlenet, resnet-18)")
opt = parser.parse_args()

# Load image and network
img = jetson_utils.loadImage(opt.filename)
net = jetson_inference.imageNet(opt.network)

# Classify the image
class_idx, confidence = net.Classify(img)
class_desc = net.GetClassDesc(class_idx)

# Print results
print(f"Image is recognized as {class_desc} (class #{class_idx}) with {confidence*100:.2f}% confidence")
</code></pre>

<ol>
    <li><b>Imports</b>: Load necessary libraries for image processing and classification.</li>
    <li><b>Arguments</b>: Parse the image filename and network type from the command line.</li>
    <li><b>Load and Classify</b>: Load the image, run it through the selected network, and print the classification result.</li>
</ol>

<h2>Training a Custom Model</h2>

<p>To train a custom model, follow these steps:</p>

<ol>
    <li><b>Prepare Dataset</b>:
        <pre><code>wget [DATASET_URL] -O dataset.tar.gz
tar xvzf dataset.tar.gz</code></pre>
    </li>
    <li><b>Configure Environment</b>:
        <pre><code>echo 1 | sudo tee /proc/sys/vm/overcommit_memory
./docker/run.sh</code></pre>
    </li>
    <li><b>Train Model</b>:
        <pre><code>python3 train.py --model-dir=models/my_model data/my_dataset</code></pre>
    </li>
    <li><b>Export Model</b>:
        <pre><code>python3 onnx_export.py --model-dir=models/my_model</code></pre>
    </li>
    <li><b>Verify Model</b>:
        <pre><code>Ensure the exported model (resnet18.onnx) is in the models/my_model directory.</code></pre>
    </li>
</ol>

<h2>Running the Trained Model</h2>

<p>To use your trained model for classification:</p>

<ol>
    <li><b>Set Up</b>:
        <pre><code>Ensure the model file is in the correct directory.</code></pre>
    </li>
    <li><b>Run Classification</b>:
        <pre><code>imagenet.py --model=models/my_model/resnet18.onnx --input_blob=input_0 --output_blob=output_0 --labels=data/my_dataset/labels.txt path/to/image.jpg</code></pre>
    </li>
</ol>

<h2>Run Local App for Detection (easy)</h2>

1. Download the exe. file <a href="https://github.com/Adam-Dalloul/ArtDetectiveAI/raw/main/Desktop_App/AI%20Image%20Classifier.exe?download=" target="_blank">here</a> and run it.
2. Find and select the .onnx model to be used and the labels.txt file to be used.
3. Upload an image to the app, it will proccess it and tell you the output of its detection and the accuracy.

<h2>Resources</h2>

<ul>
    <li><a href="https://www.kaggle.com/datasets/kausthubkannan/ai-and-human-art-classification" target="_blank">Dataset</a></li>
    <li><a href="https://github.com/dusty-nv/jetson-inference" target="_blank">Jetson Inference Documentation</a></li>
    <li><a href="https://youtube.com/" target="_blank">Video Demonstration</a></li>
</ul>

<h2>Contact</h2>
<p>For any questions or feedback, please reach out to <a href="mailto:ad@adamdalloul.com">ad@adamdalloul.com</a>.</p>

<hr/>

<p>Thank you for using ArtDetectiveAI!</p>

</body>
</html>
