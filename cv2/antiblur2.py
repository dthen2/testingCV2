import math
from scipy.special import jv
import numpy as np

# ピンボケを直すプログラムのコンボリューション関数用にベッセル関数の逆数のフーリエ変換
# smoother をかけて、スッパリmax_xで切らないようにしたバージョン

if __name__ == '__main__':

    
    dx = 0.005
    max_x = 23.
    dk = 0.02
    max_k = 12.
    max_inv = 100.
    x_band = 8.

    k = 0.
    print("With smoother. Max X=%.1f" % max_x)
    while(k < max_k):
        x = 0.
        integ = 0.
        while(x < max_x):
            y = 0.
            while(y < max_x):
                r = math.sqrt(x**2 + y**2)
                #bessel = scipy.special.jv(0, r)
                bessel = jv(0, r)
                #print("bessel = %.2f, r = %.2f" % (bessel, r))
                smoother = min(1., max(-1., (max_x - r)/x_band))
                if bessel == 0. or r > max_x:
                    inv = 0.
                else: 
                    inv = max(-max_inv,min(max_inv, 1./bessel))
                func = smoother*inv*math.cos(k*x)
                integ += func*dx*dx
                y += dx
            x += dx
        #print("%.3f, %.3f" % (k, integ))
        print("%.3f, " % integ, end ="")
        k += dk