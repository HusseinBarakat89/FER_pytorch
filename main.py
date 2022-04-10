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
            img_for_model = gray[y:y + h, x:x + w]
            img_for_model = self.transform(img_for_model).unsqueeze(0)
            with torch.no_grad():
                self.model.eval()
                pred = self.model.cpu()(img_for_model.float())
                pred = self.emotion_dict[pred.argmax(1).item()]
                cv2.putText(img, pred, ((int) (x+0.25*w), (int) (y-10)), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 20, 100))
        return img

    def read(self, source=0, show=True, output=None):
        cap = cv2.VideoCapture(source)
        if output:
            fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
            out = cv2.VideoWriter(output, fourcc, 30, (int(cap.get(3)),int(cap.get(4))))
        while cap.isOpened():
            ret, img = cap.read()
            if ret:
                img = self.detect(img)
                if show:
                    cv2.imshow('img', img)
                if output:
                    out.write(img)
            key = cv2.waitKey(30) & 0xff
            if key == 27 and source == 0 or not ret:
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    app = App()
    app.read(source=0, show=True, output=None)
