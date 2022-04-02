import torch
from torchvision import transforms
import cv2
import numpy as np

class App:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.face_cascade = cv2.CascadeClassifier('./face_detection/haarcascade_frontalface_default.xml')
        self.model = torch.jit.load('model_scripted.pt')
        self.model.eval()
        self.model.to(self.device)
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0., 255.), transforms.Resize((48, 48))])

    def detect(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            if w>0 and h>0:
                img_for_model = np.array(gray[x:x+w, y:y+h])
                cv2.imshow('img', img_for_model)
                with torch.no_grad():
                    y = self.model(self.transform(img_for_model).to(self.device)).argmax(1)
                    print(y.item())
        #cv2.imshow('img', img)

if __name__ == '__main__':
    app = App()
    cap = cv2.VideoCapture(0)
    while True:
        # Read the frame
        _, img = cap.read()
        app.detect(img)
        
        key = cv2.waitKey(30) & 0xff
        if key==27:
            break
    cap.release()




