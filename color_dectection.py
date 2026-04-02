import cv2
import numpy as np


def dectect_color(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])

    lower_red2 = np.array([170, 100, 100])
    upper_red2 = np.array([180, 255, 255])

    lower_green = np.array([35, 150, 50])
    upper_green = np.array([85, 255, 255])

    lower_blue = np.array([100, 150, 50])
    upper_blue = np.array([140, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask3 = cv2.inRange(hsv, lower_green, upper_green)
    mask4 = cv2.inRange(hsv, lower_blue, upper_blue)

    mask = mask1 + mask2 + mask3 + mask4
    cv2.imshow("Combined Mask", mask) 
    cv2.waitKey(1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    results= [] #color, x, y
    threshold = 500
    goal_pos = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > threshold:
            #Centroid
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])

                label = "UnKnown"
                if mask1[cY, cX] > 0 or mask2[cY, cX] > 0:
                    label = "Red"
                    goal_pos = [0.68, 0, 0.01]
                elif mask3[cY, cX] > 0:
                    label = "Green"
                    goal_pos = [0.48, 0.4, 0.01]
                elif mask4[cY, cX] > 0:
                    label = "Blue"
                    goal_pos = [0.48, -0.4, 0.01]
                
                results.append({"color": label, "center": [cX, cY], "area": area, "goal_pos": goal_pos})

                
                cv2.drawContours(frame, [cnt], -1, (0, 255, 0), 2)
                cv2.putText(frame, label, (cX - 20, cY - 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
        return results, frame