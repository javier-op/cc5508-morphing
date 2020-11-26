import argparse
import numpy as np
import parse

def parse_lines(lines_file_name):
    lines_i = np.empty([0, 2, 2], dtype=np.float)
    lines_g = np.empty([0, 2, 2], dtype=np.float)
    with open(lines_file_name) as lines_file:
        format_string = "{:d}: {:d}, {:d}, {:d}, {:d}, {:d}, {:d}, {:d}, {:d}"
        while True:
            line_pair = lines_file.readline()
            if not line_pair:
                break
            digits = list(parse.parse(format_string, line_pair))
            lines_i = np.append(lines_i, [[[digits[1], digits[2]], [digits[3], digits[4]]]], axis=0)
            lines_g = np.append(lines_g, [[[digits[5], digits[6]], [digits[7], digits[8]]]], axis=0)
    return lines_i, lines_g

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Morphs an image into another using Beier-Neeling's algorithm.")
    parser.add_argument('--imageA', type=str, help='Path to starting image.', required=True)
    parser.add_argument('--imageB', type=str, help='Path to resulting image.', required=True)
    parser.add_argument('-n', type=int, help='Number of steps.', required=True)
    parser.add_argument('--lines', type=str, help='Path to file with lines.', required=True)
    args = parser.parse_args()

    lines_i, lines_g = parse_lines(args.lines)