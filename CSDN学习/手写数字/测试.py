import  cv2
import numpy as np
# from sklearn.externals import joblib
import joblib

map=cv2.imread(r"C:\Users\lenovo\Desktop\One.png")
GrayImage = cv2.cvtColor(map, cv2.COLOR_BGR2GRAY)
ret,thresh2=cv2.threshold(GrayImage,127,255,cv2.THRESH_BINARY_INV)
Image=cv2.resize(thresh2,(28,28))
img_array = np.asarray(Image)
z=img_array.reshape(1,-1)

'''================================================'''

model = joblib.load('testModel'+'/model.pkl')
yPredict = model.predict(z)
print(yPredict)
y=str(yPredict)
cv2.putText(map,y, (10,20), cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255), 2, cv2.LINE_AA)
cv2.imshow("map",map)
cv2.waitKey(0)

