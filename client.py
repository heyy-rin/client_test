# import socket
#
# sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# sock.connect(('172.30.1.1', 8585)) # 서버와 연결 시도
#
#
# data = input("send massge : ")
#
# sock.send(data.encode()) # encode를 안해주면 byteSteam이 아니라고 안보내줌
#
#    # data = sock.recv(65535)
#   #  print("돌려받은 데이터 : ", data)
#

import socket
import cv2
import time
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import pygame

count = 0
#socket 수신 버퍼를 읽어서 반환하는 함수
def recvall(sock, count):
    buf = b''
    while count:
        newbuf = sock.recv(count)
        if not newbuf: return None
        buf += newbuf
        count -= len(newbuf)
    return buf

# return : camera 객체
# 카메라가 정상적으로 작동하는지 확인함
def openCamera() :

    #OpenCV를 이용해서 webcam으로 부터 이미지 추출
    capture = cv2.VideoCapture(0)

    if (capture.isOpened()):
      print("cap is open")
    else:
      print("cap error")

    return capture

# param : frame 보내려는 이미지
# return : 이미지를 배열로 만든 결과
# frame을 이미지 배열로 만들고 인코딩한다
# 이미지의 크기와 이미지 배열을 분리해서 전송한다
# 이미지 전송 후 전송된 이미지 배열을 반환

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
# initialize the FPS counter
fps = FPS().start()

# 송신을 위한 socket을 만들고 sock에 connect
def createSocket() :
    # 연결할 서버(수신단)의 ip주소와 port번호
    #TCP_IP = '172.30.1.4'
    #TCP_IP =  '192.168.0.66' '210.115.49.252'
    try :
        global TCP_IP
        global TCP_PORT
        global sock

        TCP_IP = '192.168.0.66'
        TCP_PORT = 5001
        #송신을 위한 socket 준비
        sock = socket.socket()
        sock.connect((TCP_IP, TCP_PORT))

    except socket.error as msg :
        print("socket create error ! "+ str(msg))

capture = openCamera()
personCount = 0
pygame.mixer.init()

peep = pygame.mixer.Sound("peep.wav")
shutter = pygame.mixer.Sound("shutter.wav")

while True :

    ret, frame = capture.read()
    #
    frame = imutils.resize(frame, width=600)  # window size

    # grab the frame dimensions and convert it to a blob
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (600, 600)), 0.007843, (300, 300), 127.5) # size조절은 여기서?

    # pass the blob through the network and obtain the detections and
    # predictions
    net.setInput(blob)
    detections = net.forward()

    # loop over the detections
    for i in np.arange(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence > args["confidence"]:
            # extract the index of the class label from the
            # `detections`, then compute the (x, y)-coordinates of
            # the bounding box for the object
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # draw the prediction on the frame
            label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
            if (CLASSES[idx] == "person"): # 인코딩하기 전에 사람 검출한 뒤 ! 해당되는 이미지만 보내기!

                personCount = personCount + 1
                print(personCount)
                if personCount >= 4:
                    peep.play()
                    time.sleep(1)

                cv2.rectangle(frame, (startX, startY), (endX, endY),
                              COLORS[idx], 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame, label, (startX, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

                # 전송된 이미지가 원하는 결과였는지 확인하기 위해 client에서 출력
                #data = sendImage(frame)  # 전송된 이미지배열이 저장됨!

                if personCount >= 6 :

                    createSocket() # '210.115.49.252'로 가는 socket을 생성
                    # 추출한 이미지를 String 형태로 변환(인코딩)시키는 과정
                    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
                    result, imgencode = cv2.imencode('.jpg', frame, encode_param)
                    data = numpy.array(imgencode)
                    stringData = data.tostring()
                    # 보내는 데이터 확인
                    #print("stringData "+ str(count)+ ":" , stringData)
                    # 데이터 전송
                    sock.send(str(len(stringData)).ljust(16).encode())  # 이미지 크기 먼저 저송
                    sock.send(stringData)  # 이미지 배열 전송

                    sock.close() # 데이터 전송 후 socket 닫기
                    shutter.play() # 전송 끝내고 찰칵 소리나게

                    personCount = 0

                    #time.sleep(0.8)  # 전송속도가 너무 빠르면 안됨!
                    #time.sleep(7) # 이미지 전송 후 결과 값을 받아오는 시간 기다리기? # 되받아오는 시간 필요X

            else :
                personCount = 0

                #decimg = cv2.imdecode(data, 1)  # 이미지 배열을 decode
                #cv2.imshow('CLIENT', decimg)  # decode된 이미지를 확인해서 원하는 이미지가 전송되었는지 확인

    # cv2가 저장 또는 중지를 위해 키보드 입력을 기다림
    k = cv2.waitKey(1) & 0xff # stop & save
    if (k == ord('s')):
        cv2.imwrite('sample.png', frame) # save기능은 단순 확인용
    elif (k == ord('q')):
        break

    # update the FPS counter
    fps.update()
    #cv2.destroyAllWindows()


#cv2.imwrite('sample.png', frame)

# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# window 와 socket 을 닫음
cv2.destroyAllWindows()
sock.close()