import cv2 as cv
import numpy as np

net=cv.dnn.readNet("yolov4.weights","yolov4.cfg")
classes=open("coco.names").read().strip().split("\n")

img=cv.resize(cv.imread("WhatsApp Image 2025-05-25 at 21.04.27_9c90d393 copy.jpg"),(1200,900),interpolation=cv.INTER_CUBIC)
height,width=img.shape[:2]

blob=cv.dnn.blobFromImage(img,0.00392,(416,416),True,crop=False)
net.setInput(blob)

layer_names=net.getLayerNames()
output_layers=[layer_names[i-1] for i in net.getUnconnectedOutLayers()]
outs=net.forward(output_layers)

class_ids=[]
confidences=[]
boxes=[]

for out in outs:
    for detection in out:
        scores=detection[5:]
        class_id=np.argmax(scores)
        confidence=scores[class_id]
        if confidence>0.3:
            center_x=int(detection[0]*width)
            center_y=int(detection[1]*height)
            w=int(detection[2]*width)
            h=int(detection[3]*height)
            x=int(center_x-w/2)
            y=int(center_y-h/2)
            boxes.append([x,y,w,h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

indices=cv.dnn.NMSBoxes(boxes,confidences,0.3,0.4)

for i in indices.flatten():
    x,y,w,h=boxes[i]
    label=classes[class_ids[i]]
    confidence=confidences[i]
    cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    cv.putText(img,f"{label} {confidence:.2f}",(x,y-10),cv.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)
cv.imshow("Image",img)
cv.waitKey(0)
cv.destroyAllWindows()