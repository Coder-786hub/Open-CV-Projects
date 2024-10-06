import numpy
import joblib
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np


# load model
model = joblib.load("TrafficSign_model.joblib") 


# Preprocessing Image
def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img
def equalize(img):
    img = cv2.equalizeHist(img)
    return img
def prepeocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img/255
    return img
def getCalssName(classNo):
    if   classNo == 0: return 'Speed Limit 20 km/h'
    elif classNo == 1: return 'Speed Limit 30 km/h'
    elif classNo == 2: return 'Speed Limit 50 km/h'
    elif classNo == 3: return 'Speed Limit 60 km/h'
    elif classNo == 4: return 'Speed Limit 70 km/h'
    elif classNo == 5: return 'Speed Limit 80 km/h'
    elif classNo == 6: return 'End of Speed Limit 80 km/h'
    elif classNo == 7: return 'Speed Limit 100 km/h'
    elif classNo == 8: return 'Speed Limit 120 km/h'
    elif classNo == 9: return 'No passing'
    elif classNo == 10: return 'No passing for vechiles over 3.5 metric tons'
    elif classNo == 11: return 'Right-of-way at the next intersection'
    elif classNo == 12: return 'Priority road'
    elif classNo == 13: return 'Yield'
    elif classNo == 14: return 'Stop'
    elif classNo == 15: return 'No vechiles'
    elif classNo == 16: return 'Vechiles over 3.5 metric tons prohibited'
    elif classNo == 17: return 'No entry'
    elif classNo == 18: return 'General caution'
    elif classNo == 19: return 'Dangerous curve to the left'
    elif classNo == 20: return 'Dangerous curve to the right'
    elif classNo == 21: return 'Double curve'
    elif classNo == 22: return 'Bumpy road'
    elif classNo == 23: return 'Slippery road'
    elif classNo == 24: return 'Road narrows on the right'
    elif classNo == 25: return 'Road work'
    elif classNo == 26: return 'Traffic signals'
    elif classNo == 27: return 'Pedestrians'
    elif classNo == 28: return 'Children crossing'
    elif classNo == 29: return 'Bicycles crossing'
    elif classNo == 30: return 'Beware of ice/snow'
    elif classNo == 31: return 'Wild animals crossing'
    elif classNo == 32: return 'End of all speed and passing limits'
    elif classNo == 33: return 'Turn right ahead'
    elif classNo == 34: return 'Turn left ahead'
    elif classNo == 35: return 'Ahead only'
    elif classNo == 36: return 'Go straight or right'
    elif classNo == 37: return 'Go straight or left'
    elif classNo == 38: return 'Keep right'
    elif classNo == 39: return 'Keep left'
    elif classNo == 40: return 'Roundabout mandatory'
    elif classNo == 41: return 'End of no passing'
    elif classNo == 42: return 'End of no passing by vechiles over 3.5 metric tons'


# Upload Image
def upload():
    global uploaded_image_path
    file = filedialog.askopenfilename()
    if file:
        uploaded_image_path = file
        img = Image.open(file)
        photo = img.resize((295, 295))
        photo = ImageTk.PhotoImage(photo)
        output_label.config(image = photo)
        output_label.image = photo
        upload_btn.config(text = "CHECK", command = classify)


# Function to classify the uploaded image
def classify():
    if uploaded_image_path:  # Check if an image was uploaded
        img = cv2.imread(uploaded_image_path)  # Read the image using OpenCV
        img = cv2.resize(img, (32, 32))  # Resize the image to the model's input size
        img = prepeocessing(img)  # Preprocess the image
        img = img.reshape(1, 32, 32, 1)  # Reshape the image for the model

        # Predict using the loaded model
        predictions = model.predict(img)
        classIndex = np.argmax(predictions, axis=1)[0]  # Get the class with the highest probability
        probabilityValue = np.max(predictions)

        # Display the result
        result_label.config(text=f"Detected: {getCalssName(classIndex)} with confidence {probabilityValue:.2f}")
        upload_btn.config(text = "UPLOAD", command = upload)

app = Tk()
app.geometry("800x600+100+50")
app.title("Traffic Sign Detecter")
app.config(background = "green")


heading = Label(app, text = "TRAFFIC SIGN CLASSIFIER", font = ("Robot", 25 , "bold"), bg = "green", fg = "red")
heading.pack(fill = "x", ipady = 20)

frame = Frame(app, width = 310, height = 310, bg = "black", bd = 4, relief = "groove")
frame.pack()

output_label = Label(frame, text = " ", width = 295, height = 295, bg = "black")
output_label.place(x = 0, y = 0)


upload_btn = Button(app, text = "UPLOAD", font = ("ROBOT", 25, "bold"), bg = "blue", fg = "white",bd = 4, command = upload)
upload_btn.pack(pady = 20)

result_label = Label(app, font = ("ROBOT", 15, "bold"), fg = "white", bg = "green")
result_label.pack()

app.mainloop()
