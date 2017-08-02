from vb_lib import * # TODO consider changing to disambiguate
import cv2
import sys


def main(src, sk):
    reader = VideoReader(src, read_sk=sk)
    for (count, frame, fgmask) in reader.read(get_bg=False):
        reader.write_image(frame, 'frame')

if __name__ == '__main__':
    try:
        opts, args = getopt.getopt(sys.argv[1:], 's')
        if len(args) is not 1:
            usage()
            sys.exit(1)
        main(args[0], ('-s', '') in opts)
    except getopt.GetoptError:
        usage()
        sys.exit(1)
