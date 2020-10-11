import os
import cv2
import glob
import numpy as np
import json

def cal_for_frames(video_path):
    frames = glob.glob(os.path.join(video_path, '*.jpg'))
    frames.sort()

    flow = []
    prev = cv2.imread(frames[0])
    prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    for i, frame_curr in enumerate(frames[1:]):
        print(i)
        curr = cv2.imread(frame_curr)
        curr = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
        tmp_flow = compute_TVL1(prev, curr)
        flow.append(tmp_flow)
        prev = curr

    return flow


def compute_TVL1(prev, curr, bound=15):
    """Compute the TV-L1 optical flow."""
    #TVL1 = cv2.DualTVL1OpticalFlow_create()

    # TVL1 = cv2.DualTVL1OpticalFlow_create()
    TVL1=cv2.createOptFlow_DualTVL1()
    flow = TVL1.calc(prev, curr, None)

    assert flow.dtype == np.float32

    flow = (flow + bound) * (255.0 / (2 * bound))
    flow = np.round(flow).astype(int)
    flow[flow >= 255] = 255
    flow[flow <= 0] = 0

    return flow


def save_flow(video_flows, flow_path):
    if not os.path.exists(os.path.join(flow_path, 'u')):
        os.mkdir(os.path.join(flow_path, 'u'))
    if not os.path.exists(os.path.join(flow_path, 'v')):
        os.mkdir(os.path.join(flow_path, 'v'))
    for i, flow in enumerate(video_flows):
        cv2.imwrite(os.path.join(flow_path, 'u', "{:06d}.jpg".format(i)),
                    flow[:, :, 0])
        cv2.imwrite(os.path.join(flow_path, 'v', "{:06d}.jpg".format(i)),
                    flow[:, :, 1])


def extract_flow(video_path, flow_path):
    flow = cal_for_frames(video_path)
    print(flow_path)
    save_flow(flow, flow_path)
    print('complete:' + flow_path)
    return


def getFLOW(path_to_video,path_to_save):
    flow_path=path_to_save
    if not os.path.exists(os.path.join(flow_path, 'u')):
        os.mkdir(os.path.join(flow_path, 'u'))
    if not os.path.exists(os.path.join(flow_path, 'v')):
        os.mkdir(os.path.join(flow_path, 'v'))
    return
    video_path = path_to_video  # 保存帧的路径
    save_path = path_to_save
    extract_flow(video_path, save_path)

def rgb():
    k=5
    for i in range(1,101):
        avis = glob.glob(os.path.join("./{}/".format(i), '*.avi'))
        #with open("./{}/avinum.txt".format(i),"w") as f:
            #f.write(str(len(avis)))
        avis.sort()
        for (j,avi) in enumerate(avis):
            print(str(i)+" "+str(j))
            desp="./{}/{}/".format(i,j+1)
            if (not os.path.exists(desp)):
                os.mkdir(desp)
            small_rgbpath = "./{}/{}/rgb/".format(i,j+1)
            if (not os.path.exists(small_rgbpath)):
                os.mkdir(small_rgbpath)

            cap = cv2.VideoCapture(avi)
            rate = cap.get(5)  # 获取帧率
            fraNum = cap.get(7)  # 获取帧数
            duration = fraNum / rate
            if (duration<1):
                k=1
            else:
                k=7

            success, frame = cap.read()
            imagenum=0
            t=0
            while (success):
                t+=1
                if (t%k==0):
                    imagenum+=1
                    cv2.imwrite(small_rgbpath +"/"+ str(imagenum)+ ".jpg", frame)
                # if (heng==0):
                # frame=cv2.transpose(frame)
                # if nf%100==0:
                #     print(nf)
                # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # frame = cv2.resize(frame,(int(frame.shape[1]/2),int(frame.shape[0]/2)))

                success, frame = cap.read()
            cap.release()

def flow():
    for i in range(1,101):
        avis = glob.glob(os.path.join("./{}/".format(i), '*.avi'))
        avis.sort()
        for (j,avi) in enumerate(avis):

            small_rgbpath = "./{}/{}/rgb/".format(i,j+1)
            small_flowpath = "./{}/{}/flow/".format(i, j+1)
            if (not os.path.exists(small_flowpath)):
                os.mkdir(small_flowpath)

            print(i)
            print(j)
            getFLOW(small_rgbpath,small_flowpath)

def writeJson():
    for i in range(1, 101):
        avis = glob.glob(os.path.join("./{}/".format(i), '*.avi'))

        dic={}
        for (j, avi) in enumerate(avis):
            jdic={"subset":"testing"}
            print(str(i) + " " + str(j))
            cap=cv2.VideoCapture(avi)
            if cap.isOpened():
                rate = cap.get(5)  # 获取帧率
                fraNum = cap.get(7)  # 获取帧数
                duration = fraNum / rate

            jdic['duration']=duration
            jdic['actions']=[[1,0.0,duration]]

            dic[str(j+1)]=jdic
        with open('./{}/mydataset.json'.format(i), 'w') as f:
            json.dump(dic, f)

writeJson()