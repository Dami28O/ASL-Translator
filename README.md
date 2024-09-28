# ASL TRANSLATOR
## Description

Welcome to a small project that I did to recognise and translate ASL fingerspelling into corresponding text using machine learning and computer vision techniques. My motivation behind this project lies with my experience over summer where one of my friends was trying to teach me sign language. However, I was mostly inspired by Project Euphonia, a DeepMind project which worked to give people with speech difficulties their voice back with AI. For instance those with ALS who develop speech impediments that hinders their ability to communicate, such technology would help give them their voice back. Projects like Project Euphonia and my project, are just examples of how we can utilise AI to improve the quality of life for those all over the world who are dealing with a variety of disabilities. My application relies on TensorFlow to create a MLP model based on hand landmarks attained from the google's MediaPipe open-source framework. It then uses this trained model in real time to classify and translate ASL fingerspelling gestures into text.

The model itself has an accuracy of 94% but when used in real-time has some difficulties correctly classifying certain symbols. Certain gestures work better on one hand than another and it has difficulties differentiating between similar gestures like the ASL gestures for "C" and "O". This limitation likely comes from the fact that I used a subset of the dataset to train and test my model due to the large scale, hence it is not as robust as it could be. If i was able to use all of the data, then I likely would have a better accuracy in real time. Generally, this makes the application suitable for use but it doesn't work as succinctly as it could, and there is a lot of room for improvement. I chose to use MediaPipe's framework to get landmarks on the hands because it would be less susceptible to different lightings and other issues that may arise from using a convolution neural network instead for example. Therefore using Mediapipe has added to the robustness and usability of my application. Another modification that I would like to add in future would be training the model to recognise motion and not just static images because ASL finger spelling contains motioned gestures, specifically the letters "J" and "Z". However, this current iteration is only trained on static images, hence it would not be able to accurately identify those mentioned letters repeatedly. 

## Usage
Two main files for this project, one to train the model and a second to use the model in real time. 

### Real Time Classification
1. **Run the Application**:
   To start the ASL translator:
   ```bash
   python3 LiveGesture.py
   ```

2. **Real-Time Translation**:
   The application will access your webcam and process live video. It will detect hand landmarks and classify fingerspelling gestures into letters. The predicted letter will be displayed in real-time in addition to a string of all letters produced. Note that the sentence has not been formatted for longer sentences and works best for short phrases. **The model only works for one hand at a time**,


If you'd like to train your own model, follow these steps:

1. **Data Collection**:
   Collect and label images of hand gestures for each letter. I used the [ASL alphabet dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet/data) or create your own.

2. **Preprocessing**:
   The SLMode.py file already processes the images by passing them into MediaPipe's landmarker. Currently it has 28 different categories, one for each letter of the alphabet as well as "del" and "space"

3. **Model Training**:
   To train the model you would use the code below, and pass in a directory of your dataset. Just note if using your own dataset, then you may need to configure the code to read it currently as it is currently configured to read the data from the dataset mentioned above.

   ```bash
   python3 SLmodel.py Data
   ```

# Credits

This project would not have been possible without the contributions and support from the following:
- **Google's MediaPipe Team** for providing the hand landmark detection solution used in this project.
- **Kaggle and Dataset Contributors** for the [ASL alphabet dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet/data), which was instrumental in training the model.
- **CS50’s Introduction to Artificial Intelligence** with Python for foundational AI concepts.
- Thanks to open-source contributors who maintain libraries like TensorFlow, OpenCV, and others used in this project.

## License
This project is licensed under the MIT License - see the [LICENSE](./LICENSE.txt) file for details.