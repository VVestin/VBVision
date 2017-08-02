from vb_lib import *
import cv2
import pickle
import sys


def main(src, sk):
    reader = VideoReader(src, read_sk=sk)

    best_ball = reader.load_obj('best_ball')
    
    background = reader.read_image('bg')

    #for (count, frame, fgmask) in reader.read():
    #    if count > best_ball.get_last_frame():
    #        best_ball.mean_shift()
    #    best_ball.draw(frame)

    for (count, frame, fgmask) in reader.read():
        mask = best_ball.circle_contour_mask(count, -9)

        masked_ball = cv2.bitwise_or(frame, frame, mask=mask)

        mask = cv2.bitwise_not(mask)
        masked_bg = cv2.bitwise_or(background, background, mask=mask)
        
        background = cv2.bitwise_or(masked_ball, masked_bg)
        reader.write_image(background, 'ball_only')


def usage():
    print 'Usage: %s file_name ' % (re.sub('^.*/','',sys.argv[0]))
    print '         file_name: The name of the video in the res folder without .mp4 extension'
    print '         -s: Use the skvideo vreader (default is opencv VideoCapture)'
    print ' '
    print '    Outputs only the top ball candidate moving on the background'
    print ' '
    print '\n'

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
