import os
import cv2


def writeVideo(vid, path):
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')

    (x,y)=(vid[0].shape[1],vid[0].shape[0])

    out = cv2.VideoWriter(path, fourcc, 20.0, (x,y))
    for frame in vid:
        # cv2.imshow('frame',frame)
        # cv2.waitKey(1)
        out.write(frame)
    out.release()

def loadVideo(path,key_frame,i):
    print("loading...")
    cap = cv2.VideoCapture(path)
    print(path)
    res = []
    success, frame = cap.read()

    nf = 0
    t = 1

    small_avipath = "./{}".format(i)
    if (not os.path.exists(small_avipath)):
        os.mkdir(small_avipath)

    k=0
    while (success):
        res.append(frame)
        if (nf > 0 and nf == key_frame[t]):
            k+=1
            writeVideo(res, small_avipath + "/" + str(k) + ".avi")
            t+=1
            res=[]
        nf = nf + 1
        # if (heng==0):
        # frame=cv2.transpose(frame)
        # if nf%100==0:
        #     print(nf)
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # frame = cv2.resize(frame,(int(frame.shape[1]/2),int(frame.shape[0]/2)))

        success, frame = cap.read()
    cap.release()


def split():
    for i in range(1,101):
        key_frame=[]
        with open("./output{}.txt".format(i),"r") as f:
            lines=f.readlines()
            for (t,line) in enumerate(lines):

                st=line.split(",")
                if (st[1]=="1\n" or t+1==len(lines)):
                    key_frame.append(t)
            key_frame.append(0)

        path="./{}.mp4".format(i)

        loadVideo(path,key_frame,i)

