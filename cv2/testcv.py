import cv2
import numpy as np

# http://makotomurakami.com/blog/2020/03/28/4232/

if __name__ == '__main__':
    #              B      G   R
    b = np.array([255.0, 0.0, 0.0])
    g = np.array([0.0, 255.0, 0.0])
    r = np.array([0.0, 0.0, 255.0])
    x = np.array([b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,g,g,g,g,g,g,g,g,g,g,g,g,g,g,g,g,g,g,g,g,g,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r])
    f = np.array([b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,g,g,g,g,g,g,g,g,g,g,g,g,g,g,g,g,g,g,g,g,g,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r])
    p = np.array([b,b,b,b,b,b,b,b,b,b,b,b,b,g,g,g,g,g,g,g,g,g,g,g,g,g,g,g,g,g,g,g,g,g,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r])
    y = np.array([x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p])
    cv2.imshow('test1', y)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    # put 2*2 size color points
    k = np.array([0.0, 0.0, 0.0])
    w = np.array([250.0, 250.0, 250.0])
    l = np.array([k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k  ])
    n = np.array([k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k, w,w ,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k])
    n2 =np.array([k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k, b,b, k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k])
    n3 =np.array([k,k,k,k,k,k,k,k,k,k,k,k,k,k, r,r ,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k])
    n4 =np.array([k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k,k, g,g, k,k,k,k,k,k,k,k,k,k,k,k,k,k])
    y2 =np.array([l,l,l,l,l,l,l,l,l, n,n, l,l,l,l,l,l,l,l,l,l,l, n2,n2 ,l,l,l,l,l,l,l,l,l,l,l,l,l,l, n4,n4 ,l,l,l,l,l,l,l,l,l,l,l, n3,n3, l,l,l,l,l,l,l,l,l,l,l])
    cv2.imshow('test2', y2)

    img = cv2.imread('DSC_0328_.JPG')
    cv2.imshow('nana', img)

    imgs = cv2.resize(img, (200, 130))
    cv2.imshow('nana2', imgs)

    top = 65
    bottom = 105
    left = 95
    right = 140
    imgp = imgs[top : bottom, left : right]
    cv2.imshow('nana nose', imgp)
    nose_color = imgp[39,44]
    print('nose BGR =', nose_color)

    top = 70
    bottom = 100
    left = 0
    right = 40
    imgf = imgs[top : bottom, left : right]
    cv2.imshow('floor', imgf)

    floor_color = imgf[15,20]
    print('floor BGR =', floor_color)

    cv2.waitKey(0)

    cv2.destroyAllWindows()