# 📷 Image Sentiment Analysis with CNNs  

## 🌟 Overview  
This project uses two complementary approaches for image sentiment analysis:

- Convolutional Neural Networks (CNN): Predicts sentiment from images while providing interpretability through heat maps that highlight key regions influencing the predictions.
- Image Captioning with GPT: Generates captions for images to extract contextual sentiment, leveraging language-based sentiment analysis.


![image](https://github.com/user-attachments/assets/2315deff-98ee-42c3-acf4-83bfa785c466)
![image](https://github.com/user-attachments/assets/91f87cf3-16bb-4b04-9909-6bd2d29cb7ef)

---

## 🛠️ Key Features  
- **Sentiment Analysis**: Predicts positive, negative, or neutral sentiment from input images.  
- **Heat Maps**: Visualizes image regions that impact the model's sentiment predictions the most.  
- **Model Explainability**: Improves understanding of CNN behavior and decision-making processes for better interpretability.  
- **Scalable Framework**: Designed for seamless integration with other image or video datasets.  

---

## 🧰 Technologies Used  

### **Languages**  
![Python](https://img.shields.io/badge/Python-%233776AB.svg?style=for-the-badge&logo=python&logoColor=white)

### **Frameworks and Libraries**  
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=tensorflow&logoColor=white)  
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=pytorch&logoColor=white)  
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23FF9F00.svg?style=for-the-badge&logo=python&logoColor=white)  
![OpenCV](https://img.shields.io/badge/OpenCV-%235C2D91.svg?style=for-the-badge&logo=opencv&logoColor=white)

### **Tools**  
- **Jupyter Notebooks**: For model development and visualization.  
- **NumPy & Pandas**: For data manipulation and preprocessing.  

---

## 📂 Project main components  

1. **Model Architecture**  
   - **Convolutional Neural Network (CNN)** with layers optimized for feature extraction and classification.  
   - Integrated dropout layers for regularization and improved generalization.  

2. **Heat Map Generation**  
   - Visualizes salient regions of input images using Grad-CAM or similar techniques to explain predictions.  

3. **Training & Evaluation**  
   - Dataset split into training, validation, and testing sets.  
   - Metrics include accuracy, precision, recall, and F1 score for performance evaluation.  

4. **Insights & Results**  
   - Heat maps provide actionable insights for image interpretation, especially in sentiment-sensitive applications.  

---

## 🎯 Possible Use Cases  
- **Marketing**: Understand audience sentiment from campaign images.  
- **Social Media Analytics**: Analyze and monitor public sentiment through shared visuals.  
- **Visual Storytelling**: Gain insights into emotional responses triggered by imagery.  

---

## 🚀 Getting Started  

### Prerequisites  
- Python 3.8+  
- Required libraries: TensorFlow, PyTorch, NumPy, OpenCV, Matplotlib  

### Installation  
1. Clone the repository:  
    ```bash
    git clone https://github.com/andremuna00/ImageSentimentAnalysis.git
    ```
2. Install dependencies
3. Run the notebook or script to train the model and generate heat maps.

## 📊 Results
- Accuracy: Achieved 60% on the test dataset.
- Heat Map Insights: Demonstrated accurate and interpretable sentiment predictions.
