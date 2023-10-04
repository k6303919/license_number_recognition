import torch
import ultralytics
from ultralytics import YOLO
import tkinter as tk
from tkinter import *
from PIL import Image, ImageTk
import os
from tkinter.filedialog import askopenfilename, asksaveasfilename, asksaveasfile
import cv2
import numpy as np
import pytesseract
import pyocr
import pyocr.builders
import re
file_path = os.path.dirname(__file__) 
# Fetching the latest runs
yolov5_path = '/Users/zhanjunwen/Downloads/content 2/yolov5/runs/train'
latest_run = os.listdir(yolov5_path)[-1]

# Fetching the best weights
best_weights = os.path.join(yolov5_path, latest_run, 'weights', 'best.pt')

# Loading the model with best weights trained on custom data
Yolov5 = torch.hub.load('ultralytics/yolov5', 'custom', best_weights)
Yolov8 = YOLO('/Users/zhanjunwen/Downloads/predict_plate/train5/runs/detect/train5/weights/best.pt')
class image_process():
    def __init__(self):
        self.root=tk.Tk()                                                      
        self.root.title("車牌辨識系統")                           
        self.root.geometry("800x700") 

        # 創建menu變量
        menubar = tk.Menu(self.root) # 創建menu
        self.root.config(menu=menubar)
        # 創建下拉menu 開圖檔實例
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="開檔", menu=file_menu)
        file_menu.add_command(label="打開圖檔", command=self.open_file)
        file_menu.add_command(label="圖檔清除", command=self.clear)
        # 創建下拉menu 開啟預測實例
        convert_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="預測", menu=convert_menu)
        convert_menu.add_command(label="Yolov5車牌預測", command=self.Yolov5_predict)
        convert_menu.add_command(label="Yolov8車牌預測", command=self.Yolov8_predict)
        convert_menu.add_command(label="Yolov8車牌微調", command=self.test_Yolov8_predict)

        # 創建影像容器，self.root視窗變量
        # padx pady 容器需要保留的空間
        self.frame_source = tk.LabelFrame(self.root, text="Source image:")
        self.frame_source.place(x = 1, y = 30, width = 400, height  = 400)
        # 創建第二影像容器，self.root視窗變量
        self.frame_process = tk.LabelFrame(self.root, text="Process image:")
        self.frame_process.place(x=400, y=30, width=400, height=400)
        # 創建第三影像容器，self.root視窗變量
        self.frame_predict = tk.LabelFrame(self.root, text="Detection license number:")
        self.frame_predict.place(x=1, y=450, width=400, height=200)
        # 創建第四標籤文字容器，self.root視窗變量
        self.frame_mylabel = tk.LabelFrame(self.root, text='Predict license number')
        self.frame_mylabel.place(x=400, y=450, width=400, height=200)
        # self.frame_predict.place(x=1, y=450, width=400, height=200)


        # 讀取影像轉陣列灰階
        #path = '200x200-Lenna.png'
        #image = cv2.imread(path)
        #image = Image.open(test_file_path)
        #image_gray = cv2.imdecode(np.fromfile(image), flags=cv2.COLOR_BGR2GRAY) # 讀取影像
        self.canvas_source = None
        self.canvas_process = None
        self.canvas_predict = None
        self.canvas_label = None
        self.path = ''
        self.root.mainloop()
        
    def open_file(self):
        open_img_path = askopenfilename(initialdir=file_path,
                                        filetypes=[("jpg file", "jpg"), ("png file", "png"), ("bmp file", "bmp")],
                                        parent=self.root,
                                        title="打開圖片")
        if (open_img_path == ''):
            return
        
        self.path = open_img_path
        tk_image = Image.open(self.path)
        
        test_image = ImageTk.PhotoImage(tk_image)
        if (self.canvas_source == None):
            self.canvas_source = tk.Label(self.frame_source, image=test_image, background='#aaa')
        self.canvas_source.configure(image=test_image)
        self.canvas_source.pack(expand=1)
      
        self.root.mainloop()

    def clear(self):
        if(self.canvas_source != None):
            self.canvas_source.pack_forget()
            self.canvas_source = None
            self.path = ''
        if(self.canvas_process != None):
            self.canvas_process.pack_forget()
            self.canvas_process = None
            self.path = ''
        if(self.canvas_predict != None):
            self.canvas_predict.pack_forget()
            self.canvas_predict = None
            self.path = ''

        if(self.canvas_label != None):
            self.canvas_label.grid_forget()
            self.canvas_label = None
            self.path=''
    
    def Yolov5_predict(self):
        image = cv2.imdecode(np.fromfile(self.path ,dtype=np.uint8), 1) # np.fromfile
        image2 = image.copy()
        blur = cv2.GaussianBlur(image2, (3,3), 0)
        thresh = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY)[1]
        invert = 255 - thresh
        background = invert + blur
        # Predicting from model
        results = Yolov5(background)
        dfResults = results.pandas().xyxy[0]
        # print(dfResults)
        for box in results.xyxy[0]:
            xA = int(box[0]) 
            xB = int(box[2])
            yA = int(box[1])
            yB = int(box[3])
            cv2.rectangle(image2, (xA, yA), (xB, yB), (255,0,0), 2)
            classid = dfResults.iloc[0]['name']
            xmin = round(dfResults.iloc[0]['xmin'],2)
            xmax = round(dfResults.iloc[0]['xmax'],2)
            ymin =round(dfResults.iloc[0]['ymin'],2)
            ymax =round(dfResults.iloc[0]['ymax'],2)
            cords = [xmin,xmax,ymin,ymax]
            conf = round(dfResults.iloc[0]['confidence'],2)
            print("Object type:",classid)
            print("Coordinates:",cords)
            print("Probability:",conf)
            text = "{}: {:.2f}".format(classid, conf)
            cv2.putText(image2 , text, (xA, yA - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1, 4)
            number_plate = background[yA:yB,xA:xB]
            # Getting co ordinates of license plate
            config = r'--psm 6'
            plate_text = pytesseract.image_to_string(number_plate ,config=config)
            real_txt=re.findall(r'[A-Z]+|[\d]+',plate_text)
            print(f"車牌是： {real_txt}")
            origin_plate = Image.fromarray(image2[:,:,::-1])
            origin_plate_2 = ImageTk.PhotoImage(image=origin_plate)
            if (self.canvas_process == None):
                self.canvas_process = tk.Label(self.frame_process, image=origin_plate_2)
            self.canvas_process.configure(image=origin_plate_2)
            self.canvas_process.pack(expand=1)

            predict_plate = Image.fromarray(number_plate)
            predict_plate_2 = ImageTk.PhotoImage(image=predict_plate)
            # for crop predict plate image
            if (self.canvas_predict == None):
                self.canvas_predict = tk.Label(self.frame_predict, image=predict_plate_2)
            self.canvas_predict.configure(image=predict_plate_2)
            self.canvas_predict.pack(expand=1)

            if (self.canvas_label == None):
                    self.canvas_label = tk.Label(self.frame_mylabel, text=real_txt,font=('Arial',60))
            self.canvas_label.config(text=real_txt)
            self.canvas_label.grid(row=2, column=1, ipady=50, ipadx=55)
                # self.canvas_label.pack()

            self.root.mainloop()
            return origin_plate_2, predict_plate_2, real_txt
            
    def Yolov8_predict(self):
        image = cv2.imdecode(np.fromfile(self.path ,dtype=np.uint8), 1) # np.fromfile
        image2 = image.copy()
        blur = cv2.GaussianBlur(image2, (7,7), 0)
        _,thresh = cv2.threshold(blur, 127, 255,cv2.THRESH_BINARY_INV+cv2.THRESH_BINARY) #轉為黑白
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        invert = 255 - opening
        background = invert + blur
        results = Yolov8.predict(source=background)
        for result in results:
            boxes = result.boxes.cpu().numpy()
            # print(boxes)
            for box in boxes:
                cords=box.xyxy[0].astype(int)
                class_id = result.names[box.cls[0].item()]#.replace("plate", "License Plate")
                conf = round(box.conf[0].item(), 2)
                crop = background[int(box.xyxy[0, 1]):int(box.xyxy[0, 3]), int(box.xyxy[0, 0]):int(box.xyxy[0, 2])]
                print("儲存預測邊界框")
                cv2.imwrite('pred1.png', crop)
                print("Object type:", class_id)
                print("Coordinates:", cords)
                print("Probability:", conf)
                cv2.rectangle(image2 , cords[:2], cords[2:], (0,255,0),2)
                text = "{}: {:.2f}".format(class_id, conf)
                config = r'--psm 6'
                plate_text = pytesseract.image_to_string(crop,config=config)
                real_txt=re.findall(r'[A-Z]+|[\d]+',plate_text)
                cv2.putText(image2 , text, (cords[0], cords[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1,4)
                print(f"車牌是： {real_txt}")
                origin_plate = Image.fromarray(image2[:,:,::-1])
                origin_plate_2 = ImageTk.PhotoImage(image=origin_plate)
                if (self.canvas_process == None):
                     self.canvas_process = tk.Label(self.frame_process, image=origin_plate_2)
                self.canvas_process.configure(image=origin_plate_2)
                self.canvas_process.pack(expand=1)

                predict_plate = Image.fromarray(crop)
                predict_plate_2 = ImageTk.PhotoImage(image=predict_plate)
                # for crop predict plate image
                if (self.canvas_predict == None):
                     self.canvas_predict = tk.Label(self.frame_predict, image=predict_plate_2)
                self.canvas_predict.configure(image=predict_plate_2)
                self.canvas_predict.pack(expand=1)

                if (self.canvas_label == None):
                    self.canvas_label = tk.Label(self.frame_mylabel, text=real_txt,font=('Arial',60))
                self.canvas_label.config(text=real_txt)
                self.canvas_label.grid(row=2, column=1, ipady=50, ipadx=55)
                # self.canvas_label.pack()

                self.root.mainloop()
                return origin_plate_2, predict_plate_2, real_txt
            
    def test_Yolov8_predict(self):
        # if (self.path == ''):
        #     return 
            
        image = cv2.imdecode(np.fromfile(self.path ,dtype=np.uint8), 1) # np.fromfile
        image2 = image.copy()
        # resize_image = cv2.resize(image2, (510, 300))
        blur_gray = cv2.GaussianBlur(image2,(5,5),0)
        _,thresh = cv2.threshold(blur_gray, 127, 255,cv2.THRESH_BINARY) #轉為黑白
        results = Yolov8.predict(source=thresh )
        for result in results:
            boxes = result.boxes.cpu().numpy()
            for box in boxes:
                cords=box.xyxy[0].astype(int)
                class_id = result.names[box.cls[0].item()]
                conf = round(box.conf[0].item(), 2)
                crop = thresh[int(box.xyxy[0, 1]):int(box.xyxy[0, 3]), int(box.xyxy[0, 0]):int(box.xyxy[0, 2])]
                # print("儲存預測邊界框")
                # cv2.imwrite('pred1.png', crop)
                print("Object type:", class_id)
                print("Coordinates:", cords)
                print("Probability:", conf)
                cv2.rectangle(image2, cords[:2], cords[2:], (0,255,0),2)
                text = "{}: {:.2f}".format(class_id, conf)
                config = r'--psm 6'
                # img_rgb = Image.frombytes('RGB', invert .shape[:2], invert, 'raw', 'BGR', 0, 0)
                plate_text = pytesseract.image_to_string(crop,config=config)
                real_txt=re.findall(r'[A-Z]+|[\d]+',plate_text)
                cv2.putText(image2, text, (cords[0], cords[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                # print("優化後辨識結果：",real_txt)
                print(f"車牌是： {real_txt}")
                # for origin plate image
                origin_plate = Image.fromarray(image2[:,:,::-1])
                origin_plate_2 = ImageTk.PhotoImage(image=origin_plate)
                if (self.canvas_process == None):
                    self.canvas_process = tk.Label(self.frame_process, image=origin_plate_2)
                self.canvas_process.configure(image=origin_plate_2)
                self.canvas_process.pack(expand=1)

                predict_plate = Image.fromarray(crop)
                predict_plate_2 = ImageTk.PhotoImage(image=predict_plate)
                # for crop predict plate image
                if (self.canvas_predict == None):
                    self.canvas_predict = tk.Label(self.frame_predict, image=predict_plate_2)
                self.canvas_predict.configure(image=predict_plate_2)
                self.canvas_predict.pack(expand=1)

                if (self.canvas_label == None):
                    self.canvas_label = tk.Label(self.frame_mylabel, text=real_txt,font=('Arial',60))
                self.canvas_label.config(text=real_txt)
                self.canvas_label.grid(row=2, column=1, ipady=50, ipadx=55)
                # self.canvas_label.pack()

                self.root.mainloop()
                return origin_plate_2, predict_plate_2, real_txt



if __name__ == '__main__':
    image_process()
    
