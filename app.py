import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import onnxruntime as ort
import numpy as np

class ArtDetectiveApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Image Classifier")
        self.root.geometry("600x700")  
        self.root.configure(bg="#f7f7f7")

        # Header Frame
        self.header_frame = tk.Frame(root, bg="#2c3e50", padx=20, pady=10)
        self.header_frame.pack(fill="x")

        self.title_label = tk.Label(self.header_frame, text="Welcome to AI Image Classifier",
                                    font=("Arial", 18, "bold"), bg="#2c3e50", fg="#ecf0f1")
        self.title_label.pack()

        self.subtitle_label = tk.Label(self.header_frame, text="Detect an image using a .onnx file and its corresponding labels.txt file.",
                                       font=("Arial", 12), bg="#2c3e50", fg="#bdc3c7")
        self.subtitle_label.pack()

        # Instructions Frame
        self.instructions_frame = tk.Frame(root, bg="#f7f7f7", padx=20, pady=10)
        self.instructions_frame.pack(fill="x")

        self.instructions_label = tk.Label(self.instructions_frame, text="Instructions: Follow the steps below to classify an image.",
                                           font=("Arial", 10), bg="#f7f7f7", fg="#2c3e50")
        self.instructions_label.pack()

        # Step 1: Select Model
        self.step1_label = tk.Label(root, text="Step 1: Select the ONNX Model",
                                    font=("Arial", 12, "bold"), bg="#f7f7f7", fg="#2c3e50")
        self.step1_label.pack(pady=10)

        self.select_model_button = tk.Button(root, text="Select ONNX Model",
                                             font=("Arial", 10, "bold"), bg="#3498db", fg="white",
                                             command=self.select_model, width=20, height=2)
        self.select_model_button.pack()

        # Step 2: Select Labels (hidden until step 1 is done)
        self.step2_label = tk.Label(root, text="Step 2: Select the Labels File",
                                    font=("Arial", 12, "bold"), bg="#f7f7f7", fg="#2c3e50")

        self.select_labels_button = tk.Button(root, text="Select Labels File",
                                              font=("Arial", 10, "bold"), bg="#3498db", fg="white",
                                              command=self.select_labels, width=20, height=2)

        # Step 3: Select Image (hidden until step 2 is done)
        self.step3_label = tk.Label(root, text="Step 3: Select an Image for Classification",
                                    font=("Arial", 12, "bold"), bg="#f7f7f7", fg="#2c3e50")

        self.select_image_button = tk.Button(root, text="Select Image",
                                             font=("Arial", 10, "bold"), bg="#3498db", fg="white",
                                             command=self.select_image, width=20, height=2)

        # Step 4: Classify Image (hidden until step 3 is done)
        self.classify_button = tk.Button(root, text="Classify Image",
                                         font=("Arial", 10, "bold"), bg="#e74c3c", fg="white",
                                         command=self.run_classification, width=20, height=2)

        # Variables to store file paths
        self.model_path = None
        self.labels_path = None
        self.image_path = None

    def select_model(self):
        self.model_path = filedialog.askopenfilename(title="Select ONNX Model",
                                                     filetypes=[("ONNX Model", "*.onnx")])
        if self.model_path:
            self.step2_label.pack(pady=10)
            self.select_labels_button.pack(pady=5)

    def select_labels(self):
        self.labels_path = filedialog.askopenfilename(title="Select Labels File",
                                                      filetypes=[("Labels File", "*.txt")])
        if self.labels_path:
            self.step3_label.pack(pady=10)
            self.select_image_button.pack(pady=5)

    def select_image(self):
        self.image_path = filedialog.askopenfilename(title="Select Image",
                                                     filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
        if self.image_path:
            self.classify_button.pack(pady=10)

    def run_classification(self):
        if not (self.model_path and self.labels_path and self.image_path):
            messagebox.showerror("Error", "Please complete all steps before classification.")
            return

        result = self.classify_image(self.model_path, self.labels_path, self.image_path)
        messagebox.showinfo("Classification Result", result)

    def classify_image(self, model_path, labels_path, image_path):
        try:
            # Load the ONNX model
            session = ort.InferenceSession(model_path)
            
            # Load image using Pillow
            img = Image.open(image_path).resize((224, 224))  # Adjust size as per model input
            img = np.array(img).astype('float32') / 255.0  # Normalize to 0-1 range
            img = np.transpose(img, (2, 0, 1))  # Change to CHW format (channels, height, width)
            img = np.expand_dims(img, axis=0)  # Add batch dimension

            # Run model inference
            input_blob = session.get_inputs()[0].name
            output_blob = session.get_outputs()[0].name
            result = session.run([output_blob], {input_blob: img})[0]
            
            # Get label
            class_idx = np.argmax(result)
            
            # Load labels from labels.txt
            with open(labels_path, 'r') as f:
                labels = [line.strip() for line in f.readlines()]
            
            class_desc = labels[class_idx]
            confidence = result[0][class_idx]
            
            return f"Image classified as '{class_desc}' with {confidence*100:.2f}% confidence."
        
        except Exception as e:
            return str(e)

# Create main application window
root = tk.Tk()
app = ArtDetectiveApp(root)
root.mainloop()
