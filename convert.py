import cv2
import numpy as np
import matplotlib.pyplot as plt

def rgb_to_lab(img_rgb: np.ndarray) -> np.ndarray:
    # 1. 正規化
    img_rgb = img_rgb.astype(np.float32) / 255.0

    # 2. sRGB→リニアRGB
    mask = img_rgb > 0.04045
    img_rgb_lin = np.empty_like(img_rgb)
    img_rgb_lin[mask] = ((img_rgb[mask] + 0.055) / 1.055) ** 2.4
    img_rgb_lin[~mask] = img_rgb[~mask] / 12.92

    # 3. RGB→XYZ
    M = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])
    img_xyz = np.dot(img_rgb_lin, M.T)

    # 4. XYZ→Lab
    # D65白色点
    Xn, Yn, Zn = 0.95047, 1.00000, 1.08883
    x = img_xyz[..., 0] / Xn
    y = img_xyz[..., 1] / Yn
    z = img_xyz[..., 2] / Zn

    def f(t):
        delta = 6/29
        return np.where(t > delta**3, t ** (1/3), (t / (3*delta**2)) + 4/29)

    fx = f(x)
    fy = f(y)
    fz = f(z)

    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)

    img_lab = np.stack([L, a, b], axis=-1)
    return img_lab


def lab_to_rgb(img_lab: np.ndarray) -> np.ndarray:
    # 1. Lab→XYZ
    L, a, b = img_lab[..., 0], img_lab[..., 1], img_lab[..., 2]
    fy = (L + 16) / 116
    fx = a / 500 + fy
    fz = fy - b / 200
    
    def f_inv(t):
        delta = 6/29
        return np.where(t > delta, t ** 3, 3 * delta**2 * (t - 4/29))
    
    Xn, Yn, Zn = 0.95047, 1.00000, 1.08883
    x = f_inv(fx) * Xn
    y = f_inv(fy) * Yn
    z = f_inv(fz) * Zn
    img_xyz = np.stack([x, y, z], axis=-1)

    # 2. XYZ→リニアRGB
    M_inv = np.array([[ 3.2404542, -1.5371385, -0.4985314],
                      [-0.9692660,  1.8760108,  0.0415560],
                      [ 0.0556434, -0.2040259,  1.0572252]])
    img_rgb_lin = np.dot(img_xyz, M_inv.T)
    img_rgb_lin = np.clip(img_rgb_lin, 0, 1)

    # 3. リニアRGB→sRGB
    mask = img_rgb_lin > 0.0031308
    img_rgb = np.empty_like (img_rgb_lin)
    img_rgb[mask] = 1.055 * (img_rgb_lin[mask] ** (1/2.4)) - 0.055
    img_rgb[~mask] = 12.92 * img_rgb_lin[~mask]

    # 4. スケール&クリップ
    img_rgb = np.clip(img_rgb * 255, 0, 255).astype(np.uint8)
    return img_rgb

def transfer(img_lab_source: np.ndarray, img_lab_target: np.ndarray) -> np.ndarray:
    # L, a, b成分すべてを参照画像の分布に合わせて変換
    img_lab_trans = np.empty_like(img_lab_target)
    for i in range(3):  # 0:L, 1:a, 2:b
        X_t = img_lab_target[..., i]
        X_s = img_lab_source[..., i]
        mu_t, sigma_t = np.mean(X_t), np.std(X_t)
        mu_s, sigma_s = np.mean(X_s), np.std(X_s)
        X_new = (sigma_t / sigma_s) * (X_s - mu_s) + mu_t
        if i == 0:
            # L成分は0-100にクリップ
            X_new = np.clip(X_new, 0, 100)
        img_lab_trans[..., i] = X_new
    return img_lab_trans

def calculate_b_from_median(img_lab: np.ndarray) -> float:
    """
    a+b の中央値の絶対偏差（MAD）を基にスケールパラメータ b を計算
    """
    ab = img_lab[..., 1] + img_lab[..., 2]
    median_ab = np.median(ab)
    mad = np.median(np.abs(ab - median_ab))  # 中央値の絶対偏差
    b = mad / 0.6745  # 正規分布に基づくスケール変換
    return b

def weighted_transfer(img_lab_source: np.ndarray, img_lab_target: np.ndarray) -> np.ndarray:
    """
    色が白に近いほど、またa+bがゼロに近いほど色転送の影響を小さくする色転送
    """
    img_lab_trans = np.empty_like(img_lab_target)
    b = calculate_b_from_median(img_lab_source)  # スケールパラメータを自動計算
    for i in range(3):  # 0:L, 1:a, 2:b
        X_t = img_lab_target[..., i]
        X_s = img_lab_source[..., i]
        mu_t, sigma_t = np.mean(X_t), np.std(X_t)
        mu_s, sigma_s = np.mean(X_s), np.std(X_s)
        if i == 0:
            # L成分は0-100にクリップ
            X_new = (sigma_t / sigma_s) * (X_s - mu_s) + mu_t
            X_new = np.clip(X_new, 0, 100)
        else:
            # a, b成分の重みを計算 (a,bがゼロに近いほど影響を小さくする)
            ab = img_lab_source[..., i]  # a成分またはb成分
            weight = np.exp(-np.abs(ab) / b)  # ラプラス分布に基づく重み
            weight = weight * np.clip(img_lab_source[..., 0], 0, 1)  # L成分の影響を考慮
            weight = np.clip(weight, 0, 1)  # 重みを0-1にクリップ
            #weight > X 白い
            # 重み付き平均を計算
            X_tmp = (sigma_t / sigma_s) * (X_s - mu_s ) +  mu_t
            X_new = weight * X_tmp + (1 - weight) * X_t  # 重み付き平均
            X_new = X_tmp

        img_lab_trans[..., i] = X_new
    return img_lab_trans



#光源色の補正値取得
def ab_zero_mean(img_lab: np.ndarray):
    img_lab_corr = img_lab.copy()
    ab_means = []
    for i in [1, 2]:  # a, b
        mean = np.mean(img_lab[..., i])
        img_lab_corr[..., i] = img_lab[..., i] - mean
        ab_means.append(mean)
    return img_lab_corr, ab_means

#光源色の補正値復元
#ab_meansはab_zero_meanで取得した値
def ab_restore_mean(img_lab: np.ndarray, ab_means):
    img_lab_restored = img_lab.copy()
    for i, mean in zip([1, 2], ab_means):
        img_lab_restored[..., i] = img_lab[..., i] + mean
    return img_lab_restored

# ヒストグラムをプロットする関数
def plot_histograms_l(img_1: np.ndarray, img_2: np.ndarray, filename: str = "histogram_l.png"):
    # Lab画像のL成分
    brightness_A = img_1[..., 0]
    brightness_B = img_2[..., 0]
    # ヒストグラムをプロット
    plt.figure(figsize=(10, 5))
    plt.hist(brightness_A.ravel(), bins=256, color='blue', alpha=0.5, label='Source')
    plt.hist(brightness_B.ravel(), bins=256, color='orange', alpha=0.5, label='Transferred')
    plt.title('Brightness Histogram Comparison')
    plt.xlabel('Brightness')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig(filename)  # グラフを保存
    plt.close()  # プロットを閉じる

def plot_histograms_a(img_1: np.ndarray, img_2: np.ndarray, filename: str = "histogram_a.png"):
    # Lab画像のa成分
    brightness_A = img_1[..., 1]
    brightness_B = img_2[..., 1]
    # ヒストグラムをプロット
    plt.figure(figsize=(10, 5))
    plt.hist(brightness_A.ravel(), bins=256, color='blue', alpha=0.5, label='Source')
    plt.hist(brightness_B.ravel(), bins=256, color='orange', alpha=0.5, label='Transferred')
    plt.title('a Component Histogram Comparison')
    plt.xlabel('a Component')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig(filename)  # グラフを保存
    plt.close()  # プロットを閉じる

def plot_histograms_b(img_1: np.ndarray, img_2: np.ndarray, filename: str = "histogram_b.png"):
    # Lab画像のb成分
    brightness_A = img_1[..., 2]
    brightness_B = img_2[..., 2]
    # ヒストグラムをプロット
    plt.figure(figsize=(10, 5))
    plt.hist(brightness_A.ravel(), bins=256, color='blue', alpha=0.5, label='Source')
    plt.hist(brightness_B.ravel(), bins=256, color='orange', alpha=0.5, label='Transferred')
    plt.title('b Component Histogram Comparison')
    plt.xlabel('b Component')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig(filename)  # グラフを保存
    plt.close()  # プロットを閉じる
# L成分の範囲を広げる関数
def expand_l_range(img_lab: np.ndarray, img_lab_source: np.ndarray) -> np.ndarray:
    """
    Lab画像のL成分の範囲を広げる関数
    :param img_lab: Lab画像 (NumPy配列)
    :param img_lab_source: 参照用のLab画像 (NumPy配列)
    :return: L成分の範囲を広げたLab画像
    """
    img_lab_expanded = img_lab.copy()
    L = img_lab[..., 0]
    

    L_source = img_lab_source[..., 0]
    new_min, new_max = np.min(L_source), np.max(L_source)
    # L成分を正規化して新しい範囲にスケーリング
    L_min, L_max = np.min(L), np.max(L)
    L_scaled = (L - L_min) / (L_max - L_min) * (new_max - new_min) + new_min

    # 更新されたL成分をLab画像に適用
    img_lab_expanded[..., 0] = L_scaled
    return img_lab_expanded

def equalize_l_histogram(img_lab: np.ndarray) -> np.ndarray:
    """
    Lab画像のL成分にヒストグラム均等化を適用
    """
    img_lab_equalized = img_lab.copy()
    L = img_lab[..., 0]
    L_equalized = cv2.equalizeHist(L.astype(np.uint8))
    img_lab_equalized[..., 0] = L_equalized.astype(np.float32)
    return img_lab_equalized

def scale_ab_range(img_lab: np.ndarray, scale_factor: float = 1.2) -> np.ndarray:
    """
    Lab画像のa, b成分をスケーリング（黒と白の両方で影響を小さくする）
    :param img_lab: Lab画像
    :param scale_factor: スケールファクター
    :return: スケーリングされたLab画像
    """
    img_lab_scaled = img_lab.copy()
    L = img_lab[..., 0]  # L成分（明るさ）

    # L成分に基づく重みを計算（黒と白の両方で重みを小さくする）
    weight = 1 - np.abs((L - 50) / 50)  # L=50で最大、L=0またはL=100で最小
    weight = np.clip(weight, 0, 1)  # 重みを0-1にクリップ

    # a, b成分をスケーリング（重みを適用）
    img_lab_scaled[..., 1] = img_lab[..., 1] * (1 + (scale_factor - 1) * weight)  # a成分
    img_lab_scaled[..., 2] = img_lab[..., 2] * (1 + (scale_factor - 1) * weight)  # b成分

    # a, b成分の範囲を-128から127にクリップ
    img_lab_scaled[..., 1] = np.clip(img_lab_scaled[..., 1], -128, 127)
    img_lab_scaled[..., 2] = np.clip(img_lab_scaled[..., 2], -128, 127)
    return img_lab_scaled




if __name__ == "__main__":
    img_source = cv2.imread("/Users/takenoha/Documents/UV/opencv/img/test/img04.png")
    img_target = cv2.imread("/Users/takenoha/Documents/UV/opencv/img/test/img05.png")
    img_s_lab = rgb_to_lab(img_source)
    img_t_lab = rgb_to_lab(img_target)
    
    # 色被り除去＆記録
    img_s_lab_corr, ab_means = ab_zero_mean(img_s_lab)
    img_t_lab_corr, ab_means2 = ab_zero_mean(img_t_lab)
    
    # 色変換
    img_lab_trans = transfer(img_s_lab_corr, img_t_lab_corr)
    img_lab_trans_weighted = weighted_transfer(img_s_lab_corr, img_t_lab_corr)
    
    # L成分の範囲を広げる
    img_lab_trans_expanded = expand_l_range(img_lab_trans, img_t_lab)

    # 色被り復元
    img_lab_restored = ab_restore_mean(img_lab_trans_expanded, ab_means2)
    img_rgb_trans = lab_to_rgb(img_lab_restored)
    
    # 画像を保存する前にuint8に変換
    img_rgb_trans_uint8 = np.clip(img_rgb_trans, 0, 255).astype(np.uint8)
    cv2.imwrite("image_eL_co.jpg", img_rgb_trans_uint8)
    
    # defaultの色変換
    img_lab_trans_default = transfer(img_s_lab, img_t_lab)
    img_rgb_trans_default = lab_to_rgb(img_lab_trans_default)
    img_rgb_trans_default_uint8 = np.clip(img_rgb_trans_default, 0, 255).astype(np.uint8)
    cv2.imwrite("image_default.jpg", img_rgb_trans_default_uint8)

    # L拡張
    img_lab_trans_nocorr_expanded = expand_l_range(img_lab_trans_default, img_t_lab)
    img_rgb_trans_nocorr = lab_to_rgb(img_lab_trans_nocorr_expanded)
    img_rgb_trans_nocorr_uint8 = np.clip(img_rgb_trans_nocorr, 0, 255).astype(np.uint8)
    cv2.imwrite("image_eL.jpg", img_rgb_trans_nocorr_uint8)
    
    # a, b成分のスケーリングを適用
    img_lab_scaled = scale_ab_range(img_lab_restored, scale_factor=1.5)
    img_rgb_scaled = lab_to_rgb(img_lab_scaled)
    cv2.imwrite("image_eL_co_ab.jpg", img_rgb_scaled)
    
    # #all
    # #l拡張
    # img_lab_trans_weighted_expanded = expand_l_range(img_lab_trans_weighted, new_min=0, new_max=100)
    # # 色被り復元
    # img_lab_restored_weighted = ab_restore_mean(img_lab_trans_weighted_expanded, ab_means2)
    # # a, b成分のスケーリングを適用
    # img_lab_scaled_weighted = scale_ab_range(img_lab_restored_weighted, scale_factor=1.5)
    # img_rgb_scaled_weighted = lab_to_rgb(img_lab_scaled_weighted)
    # img_rgb_scaled_weighted_uint8 = np.clip(img_rgb_scaled_weighted, 0, 255).astype(np.uint8)
    # cv2.imwrite("image_eL_co_ab_weighted.jpg", img_rgb_scaled_weighted_uint8)
    # im = lab_to_rgb(img_lab_trans_weighted)
    # im = np.clip(im, 0, 255).astype(np.uint8)
    # cv2.imwrite("weighted.jpg", im)
    
    plot_histograms_l(img_lab_trans,img_lab_scaled)
    plot_histograms_a(img_lab_trans,img_lab_scaled)
    plot_histograms_b(img_lab_trans,img_lab_scaled)
    
    # defalt vs all(not weighted)
    # img_lab_trans,img_lab_scaled
    # img_lab_scaled_weighted,ing_lab_scaled