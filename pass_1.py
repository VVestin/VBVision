from vb_lib import * # TODO consider changing to disambiguate
import cv2
import pickle
import sys

logger = logging.getLogger('vb_logger');

net_lines = {
    "Sample1": ((674, 649), (1158, 727)),
    "Sample3": ((663, 673), (1167, 671)),
    "Sample4": ((806, 656), (1451, 633)),
    "Sample5": ((392, 575), (1120, 535))
}


def main(src, sk):
    line = None
    #for src_name in net_lines:
    #    if src.startswith(src_name):
    #        line = point_slope_form(net_lines[src_name])
    #        break

    reader = VideoReader(src, read_sk=sk)

    objs = []
    ball_candidates = []

    # First pass finds and tracks contours, evaluates which contour most resembles a ball
    for (count, frame, fgmask) in reader.read():
        if line is not None:
            cv2.line(frame, line[0], line[1], HSV_COLOR['ORANGE'], 3)

        _, contours, _ = cv2.findContours(fgmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        new_objs = select_contours(frame, contours, 25, count, line)
        objs = pair_contours(objs, new_objs, count)
        for idx, obj in reversed(list(enumerate(objs))): # Iterates backwards b/c removals
            obj.draw_contour(frame)
            if not obj.is_trackable():
                confidence = obj.is_ball()
                objs.pop(idx)

                if confidence > 0:
                    ball_candidates.append((confidence, obj))

        reader.write_image(frame, 'pass_1')
        reader.write_image(fgmask, 'fgmask')

    # Output the background only image
    bg = reader.get_background()
    reader.write_image(bg, 'bg')

    # Check all remaining contours
    for obj in objs:
        confidence = obj.is_ball()
        if confidence > 0:
            ball_candidates.append((confidence, obj))

    best_balls = concatenate_ball_candidates(ball_candidates)
    best_balls = best_balls[:1]
    for ball in best_balls:
        ball[1].draw(bg)
    reader.write_image(bg, 'best_balls')

    best_ball = best_balls[0]
    reader.dump_obj(best_ball[1], 'best_ball')


def point_slope_form(line):
    return line[0], line[1], float(line[1][1] - line[0][1]) / (line[1][0] - line[0][0])


def get_vbnet(image):
    image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    pos_1 = None
    pos_2 = None
    def mouse_handler(evt, x, y, flags, params):
        if evt is cv2.EVENT_LBUTTONDOWN:
            logger.info(pos_1)
            pos_1 = (x,y)
            logger.info(pos_1)

    cv2.namedWindow('net')
    cv2.showimage('net', image)    
    cv2.setMouseCallback('net', mouse_handler)
    while pos_1 is None or pos_2 is None:
        if cv2.waitKey(20) & 0xFF == 27:
            break
    return pos_1, pos_2

def usage():
    print 'Usage: %s file_name ' % (re.sub('^.*/','',sys.argv[0]))
    print '         file_name: The name of the video in the res folder without .mp4 extension'
    print '         -s: Use the skvideo vreader (default is opencv VideoCapture)'
    print ' '
    print '    Track contours and select the best ball candidate'
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
