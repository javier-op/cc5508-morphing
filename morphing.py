import argparse
import numpy as np
from pai_io import imread, imsave
from PIL import Image
from skimage.draw import line
import time

def parse_lines(lines_file_name):
    lines_i = np.empty([0, 2, 2], dtype=np.float32)
    lines_g = np.empty([0, 2, 2], dtype=np.float32)
    with open(lines_file_name) as lines_file:
        while True:
            line_pair = lines_file.readline()
            if not line_pair:
                break
            line_pair = line_pair.split(': ')[1]
            digits = line_pair.split(', ')
            digits = [int(x) for x in digits]
            lines_i = np.append(lines_i, [[[digits[0], digits[1]], [digits[2], digits[3]]]], axis=0)
            lines_g = np.append(lines_g, [[[digits[4], digits[5]], [digits[6], digits[7]]]], axis=0)
    return lines_i, lines_g

def draw_lines_in_image(img, lines, color):
    output = img.copy()
    for l in lines:
        xx, yy = line(int(l[0][0]), int(l[0][1]), int(l[1][0]), int(l[1][1]))
        output[xx, yy] = color
    return output

def lines_to_time_t(lines_i, linesg, t):
    return (1-t) * lines_i + t * lines_g

def perp(point):
    return np.array([point[1], -point[0]], dtype=np.float32)

def dist(point, line):
    return np.abs(np.cross(line[1]-line[0], point-line[0])/np.linalg.norm(line[1]-line[0]))

def transform_point(point, l, l_prime):
    p, q = l
    p_prime, q_prime = l_prime
    u = np.dot(point-p, q-p)/np.linalg.norm(q-p)
    v = np.dot(point-p, perp(q-p))/np.linalg.norm(q-p)
    return p_prime + u*(q_prime-p_prime)/np.linalg.norm(q-p) +\
           v*perp(q_prime-p_prime)/np.linalg.norm(q_prime-p_prime)

def bilinear_interpolation(point, origin_image, fill_color):
    max_x = origin_image.shape[0] - 1
    max_y = origin_image.shape[1] - 1
    x0 = int(np.floor(point[0]))
    x1 = x0 + 1
    a1 = point[0] - x0
    a0 = 1 - a1
    y0 = int(np.floor(point[1]))
    y1 = y0 + 1
    b1 = point[1] - y0
    b0 = 1 - b1
    if x0 < 0 or x1 > max_x:
        return fill_color
    if y0 < 0 or y1 > max_y:
        return fill_color
    result = origin_image[x0, y0]*a0*b0 + origin_image[x1, y0]*a1*b0 +\
             origin_image[x0, y1]*a0*b1 + origin_image[x1, y1]*a1*b1
    result /= (a0*b0 + a1*b0 + a0*b1 + a1*b1)
    return result

def warping_multi_lines(lines_g, lines_i, img_i, fill_color):
    max_i = img_i.shape[0]
    max_j = img_i.shape[1]
    img_g = np.zeros(img_i.shape)
    for i in range(max_i):
        for j in range(max_j):
            dx = np.zeros(2)
            w_accum = 0
            x_g = np.array([i, j])
            for k in range(len(lines_g)):
                x_i = transform_point(x_g, lines_g[k], lines_i[k])
                dx_i = x_i - x_g
                d = dist(x_g, lines_g[k])
                w = (1. / (0.001 + d)) ** 2
                dx += w * dx_i
                w_accum += w
            dx /= w_accum
            x_i = x_g + dx
            img_g[i][j] = bilinear_interpolation(x_i, img_i, fill_color)
    return img_g.astype(np.uint8)

def blending(t, img_i, img_g):
    return ((1 - t) * img_i.astype(np.float32) + t * img_g.astype(np.float32)).astype(np.uint8)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Morphs an image into another using Beier-Neeling's algorithm.")
    parser.add_argument('--imageA', type=str, help='Path to starting image.', required=True)
    parser.add_argument('--imageB', type=str, help='Path to resulting image.', required=True)
    parser.add_argument('-n', type=int, help='Number of steps.', required=True)
    parser.add_argument('--lines', type=str, help='Path to file with lines.', required=True)
    args = parser.parse_args()

    img_i = imread(args.imageA)
    img_g = imread(args.imageB)
    lines_i, lines_g = parse_lines(args.lines)

    img_i_with_lines = draw_lines_in_image(img_i, lines_i, (255, 0, 0))
    imsave('lines_origin.png', img_i_with_lines)
    img_g_with_lines = draw_lines_in_image(img_g, lines_g, (255, 0, 0))
    imsave('lines_destination.png', img_g_with_lines)
    
    dt = 1. / (args.n + 1)
    t = 0.

    imgs = []
    print('Starting image generation, may take a while.')
    for i in range(args.n):
        t += dt
        start = time.time()
        lines_t = lines_to_time_t(lines_i, lines_g, t)
        warp1 = warping_multi_lines(lines_t, lines_i, img_i, (0, 0, 0))
        warp2 = warping_multi_lines(lines_t, lines_g, img_g, (0, 0, 0))
        morph = blending(t, warp1, warp2)
        end = time.time()
        print('Image {} of {} generated, took {} seconds.'.format(i+1, args.n, end-start))
        imgs.append(morph)
    
    gif_array = [img_i]*4 + imgs + [img_g]*4 + imgs[::-1]
    gif_array = [Image.fromarray(img) for img in gif_array]
    gif_array[0].save(fp='morphing.gif', format='GIF', append_images=gif_array[1:], save_all=True, duration=200, loop=0)
