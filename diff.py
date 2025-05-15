import random
import numpy as np
import matplotlib.pyplot as plt
import random as rand
color= np.zeros((3,3))
for i in range (3):
    color[i] = [int(rand.random()*255),int(rand.random()*255),int(rand.random()*255)]
print(color)

def diff_rgb(color):
    rgb = np.zeros(2)
    for i in range(2):
        rgb[i]=np.linalg.norm(color[i] - color[i+1])
    print("rgb"+str(rgb))
def diff_hsv(color):
    diff_hsv = np.zeros(2)
    hsv = np.zeros((3,3))
    for i in range (3):
        R = color[i][0]
        G = color[i][1]
        B = color[i][2]
        
    # Normalize RGB values to 0-1 range
        var_R = R / 255.0
        var_G = G / 255.0
        var_B = B / 255.0
        
        # Find the min and max values
        var_Min = min(var_R, var_G, var_B)
        var_Max = max(var_R, var_G, var_B)
        del_Max = var_Max - var_Min
        
        # Set V (Value) to max RGB value
        V = var_Max
        
        # If there is no color (gray)
        if del_Max == 0:
            H = 0
            S = 0
        else:
            # Calculate Saturation
            S = del_Max / var_Max
            
            # Calculate the difference values
            del_R = (((var_Max - var_R) / 6) + (del_Max / 2)) / del_Max
            del_G = (((var_Max - var_G) / 6) + (del_Max / 2)) / del_Max
            del_B = (((var_Max - var_B) / 6) + (del_Max / 2)) / del_Max
            
            # Calculate Hue based on the max value
            if var_R == var_Max:
                H = del_B - del_G
            elif var_G == var_Max:
                H = (1 / 3) + del_R - del_B
            elif var_B == var_Max:
                H = (2 / 3) + del_G - del_R
            
            # Adjust Hue to be within 0 and 1
            if H < 0:
                H += 1
            if H > 1:
                H -= 1
        hsv[i] = [H, S, V]
    for i in range(2):
        diff_hsv[i] =np.linalg.norm(hsv[i] - hsv[i+1])
    print('hsv'+str(diff_hsv))
def rgb2xyz(color):
    xyz = np.zeros((3,3))
    for i in range(3):
        sR = color[i][0]
        sG = color[i][1]
        sB = color[i][2]
        var_R = sR / 255.0
        var_G = sG / 255.0
        var_B = sB / 255.0
        
        # Apply the gamma correction (for standard RGB)
        if var_R > 0.04045:
            var_R = ((var_R + 0.055) / 1.055) ** 2.4
        else:
            var_R = var_R / 12.92
            
        if var_G > 0.04045:
            var_G = ((var_G + 0.055) / 1.055) ** 2.4
        else:
            var_G = var_G / 12.92
            
        if var_B > 0.04045:
            var_B = ((var_B + 0.055) / 1.055) ** 2.4
        else:
            var_B = var_B / 12.92
            
        # Scale to [0, 100]
        var_R *= 100
        var_G *= 100
        var_B *= 100
        
        # Convert to XYZ using the standard D65/2° illuminant
        X = var_R * 0.4124 + var_G * 0.3576 + var_B * 0.1805
        Y = var_R * 0.2126 + var_G * 0.7152 + var_B * 0.0722
        Z = var_R * 0.0193 + var_G * 0.1192 + var_B * 0.9505
        
        
        xyz[i] = [X, Y, Z]
    return xyz


def diff_lab(xyz):
    lab = np.zeros((3,3))
    labb= np.zeros(2)
    for i in range(3):
        X = xyz[i][0]
        Y = xyz[i][1]
        Z = xyz[i][2]
        # Reference values for the D65/2° standard illuminant
        ref_X = 95.047
        ref_Y = 100.000
        ref_Z = 108.883
        
        # Normalize the XYZ values
        var_X = X / ref_X
        var_Y = Y / ref_Y
        var_Z = Z / ref_Z
        
        # Apply the transformation for each component
        if var_X > 0.008856:
            var_X = var_X ** (1 / 3)
        else:
            var_X = (7.787 * var_X) + (16 / 116)
            
        if var_Y > 0.008856:
            var_Y = var_Y ** (1 / 3)
        else:
            var_Y = (7.787 * var_Y) + (16 / 116)
            
        if var_Z > 0.008856:
            var_Z = var_Z ** (1 / 3)
        else:
            var_Z = (7.787 * var_Z) + (16 / 116)
        
        # Calculate CIE L*, a*, b* values
        L = (116 * var_Y) - 16
        a = 500 * (var_X - var_Y)
        b = 200 * (var_Y - var_Z)
        lab[i] = [L, a, b]
    for i in range(2):
        labb[i]=np.linalg.norm(lab[i] - lab[i+1])
    print('lab'+str(labb))


def show_and_save_colors(color, filename="colors.png"):
    """
    3つのRGB色をmatplotlibで表示＆保存。
    RGBは(255, 255, 255)形式で受け取り、画像を保存。
    """
    color1 = color[0]
    color2 = color[1]
    color3 = color[2]
    # RGB値を0〜1に正規化
    colors = [np.array(color) / 255.0 for color in [color1, color2, color3]]

    fig, ax = plt.subplots(1, 3, figsize=(6, 2))  # 3色を並べて表示
    for i in range(3):
        ax[i].imshow(np.ones((10, 10, 3)) * colors[i][None, None, :])
        ax[i].axis("off")
        ax[i].set_title(f"Color {i+1}\n{color1 if i==0 else color2 if i==1 else color3}", fontsize=10)

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    #plt.show()
    print(f"保存しました: {filename}")


# 📷 表示＆保存
diff_rgb(color)
diff_hsv(color)
xyz = rgb2xyz(color)
diff_lab(xyz)
show_and_save_colors(color)