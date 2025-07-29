from ultralytics import YOLO
import cv2
import cvzone

model=YOLO("best.pt")

cap=cv2.VideoCapture("Great Putting Drill to Show Different Lines Based on Speed shorts golf.mp4")
cap.set(3,480)
cap.set(4,480)

ball_center = None
hole_box = None


frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
temp="detection.mp4"
out = cv2.VideoWriter(temp, fourcc,fps=30, frameSize=(frame_w,frame_h))

data=list()
for i in model.names.values():
   data.append(i)

while True:
   ret,frame=cap.read()
   result=model.track(frame,stream=True,persist=True)
   for r in result:
      boxes=r.boxes
      for box in boxes:
       x1,y1,x2,y2=map(int,box.xyxy[0])
       w,h=x2-x1,y2-y1
       cvzone.cornerRect(frame,(x1,y1,w,h))
       confi=int(box.conf[0]*100)/100
       cls=int(box.cls[0])
       cls_name=data[cls]
       cvzone.putTextRect(frame, f"{cls_name} {confi:.2f}", (x1, y1),colorT=(0, 255, 0),border=0,thickness=2,colorR=None,scale=1.5)
       if cls ==0: #golf_ball
           ball_center = ((x1 + x2) / 2, (y1 + y2) / 2)
       elif  cls == 1:  # golf_hole
           hole_box = (x1, y1, x2, y2)
        
       if ball_center and hole_box:
          hx1, hy1, hx2, hy2 = hole_box
          if hx1 <= ball_center[0] <= hx2 and hy1 <= ball_center[1] <= hy2:
             cv2.putText(frame, "The putt is achieved", (50, 60),cv2.FONT_HERSHEY_SIMPLEX, 
                        1.0, (0, 255, 0), 2)
           
      annotated_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)       
      cv2.imshow("image",annotated_frame)
      out.write(frame)

   if cv2.waitKey(1) & 0xFF == 27:
       break

cap.release()
out.release()
cv2.destroyAllWindows()