import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import warnings
warnings.filterwarnings("error")


def hieu_chuan(short_img, long_img, save_name):
    # Anh moi
    short_img_new = short_img.copy()

    # Lap lai tren tat ca cac kenh
    for channel in [0,1,2]:
        channel_short = short_img[:,:,channel].copy()
        channel_long = long_img[:,:,channel].copy()
        channel_short_new = channel_short.copy()
        plt.figure(channel)
        comparagram, _, _, _ = plt.hist2d(channel_short.flatten(), channel_long.flatten(), bins=256, range=[[0, 255], [0, 255]])

        # So sanh smooth 
        comparagram_smooth = cv2.GaussianBlur(comparagram, (5, 5), 0)

        # Cat bot cac ngoai le
        comparagram_smooth[comparagram_smooth < 0.01 * np.max(comparagram_smooth)] = 0

        # Uoc tinh BTF
        selected_xis = []
        selected_yis = []
        X = np.arange(256)
        p = comparagram_smooth.copy()
        ct = 0
        for ii in range(256):
            a = np.sum(X.T @ p[ii, :]) 
            b = np.sum(p[ii, :])
            try:
                xi = a / b
                selected_xis.append(ii)
                selected_yis.append(xi)

            except RuntimeWarning:
                ct += 1

        if len(selected_xis) > 0:
            # fit to polynomial 
            X = np.array(selected_xis)
            y = np.array(selected_yis)

            i = 0
            while y[i] > y[i + 1]:
                i += 1

            i -= 1
            while i >= 0:
                y[i] = y[i + 1] - 1
                i -= 1

            degree = 5
            X_poly = PolynomialFeatures(degree=degree).fit_transform(X.reshape(-1, 1))
            lin_reg = LinearRegression()
            lin_reg.fit(X_poly, y)
            
            # Duong cong moi
            X_new = np.linspace(0, np.max(X), num=np.int(np.max(X))).reshape(-1, 1)
            X_new_poly = PolynomialFeatures(degree=degree).fit_transform(X_new)
            y_new = lin_reg.predict(X_new_poly)
            y_new[y_new < 0] = 0

            # y tang don dieu
            for ii in range(1, len(y_new)):
                if y_new[ii] < y_new[ii-1]:
                    # Khop tuyen tinh bang 10 diem truoc do
                    if ii < 10:
                        Y = y_new[:ii]
                        X_fit = X_new[:ii]
                        lin_reg = LinearRegression()
                        lin_reg.fit(X_fit.reshape(-1, 1), Y)
                        y_new[ii] = lin_reg.predict(X_new[ii].reshape(-1, 1))
                    else:
                        Y = y_new[ii-10:ii]
                        X_fit = X_new[ii-10:ii]
                        lin_reg = LinearRegression()
                        lin_reg.fit(X_fit.reshape(-1, 1), Y)
                        y_new[ii] = lin_reg.predict(X_new[ii].reshape(-1, 1))
            y_new[y_new > 255] = 255
            y_new = y_new.astype(int)
            
            # Luu plot (x_new, y_new)
            cls = {
                0: 'blue',
                1: 'green',
                2: 'red'
            }

            plt.plot(X_new, y_new, color=cls[channel])
            plt.savefig('output/comparagram_' + '_' + str(cls[channel]) + '.jpg')
            plt.close()

            # Thay doi channel_short[i][j] to y[channel_short[i][j]]
            for ii in range(channel_short.shape[0]):
                for jj in range(channel_short.shape[1]):
                    try:
                        channel_short_new[ii][jj] = y_new[channel_short[ii][jj]]
                    except IndexError:
                        pass

        # Thay the kenh moi bang short_img
        short_img_new[:,:,channel] = channel_short_new

    # Luu
    cv2.imwrite('output/' + save_name, short_img_new)
    return short_img_new
