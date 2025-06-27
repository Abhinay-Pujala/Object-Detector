import cv2 as cv
import numpy as np

def load_yolo():
    net=cv.dnn.readNet("yolov4.weights","yolov4.cfg")
    classes=open("coco.names").read().strip().split("\n")
    layer_names=net.getLayerNames()
    output_layers=[layer_names[i-1] for i in net.getUnconnectedOutLayers()]
    return net,classes,output_layers


def yolo_detection(video_source=0):
    cap=cv.VideoCapture(video_source)

    net,classes,output_layers=load_yolo()

    while True:
        ret,frame=cap.read()
        if not ret:
            break
        

        frame=cv.resize(frame,(640,480),interpolation=cv.INTER_CUBIC)
        height,width=frame.shape[:2]

        blob=cv.dnn.blobFromImage(frame,0.00392,(416,416),True,crop=False)
        net.setInput(blob)

        
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
                    class_ids.append(class_id)
                    confidences.append(float(confidence))

        indices=cv.dnn.NMSBoxes(boxes,confidences,0.3,0.4)

        for i in indices.flatten():
            x,y,w,h=boxes[i]
            label=classes[class_ids[i]]
            confidence=confidences[i]
            cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            cv.putText(frame,f"{label} {confidence:.2f}",(x,y-10),cv.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)

        cv.imshow("YOLO Webcam",frame)
        if cv.waitKey(1) & 0xFF==ord('q'):
            break
    cap.release()
    cv.destroyAllWindows()



if __name__=="__main__":
    yolo_detection()