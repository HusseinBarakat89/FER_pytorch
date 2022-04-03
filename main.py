import torch
import torchvision.transforms as transforms
import cv2


class App:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier('./face_detection/haarcascade_frontalface_default.xml')
        self.model = torch.jit.load('model_scripted.pt')
        self.model.float()
        self.emotion_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((48, 48))])

    def detect(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(img)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            img_for_model = gray[y:y + h, x:x + w] / 255
            img_for_model = self.transform(img_for_model).unsqueeze(0)
            with torch.no_grad():
                self.model.eval()
                y = self.model.cpu()(img_for_model.float())
                pred = self.emotion_dict[y.argmax(1).item()]
                print(pred)
        cv2.imshow('img', img)



if __name__ == '__main__':
    app = App()
    cap = cv2.VideoCapture(0)
    while True:
        # Read the frame
        _, img = cap.read()
        app.detect(img)

        key = cv2.waitKey(30) & 0xff
        if key == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
