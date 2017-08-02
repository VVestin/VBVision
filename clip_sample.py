from vb_lib import * # TODO consider changing to disambiguate
import cv2
import getopt
import pickle
import re
import sys
# Pad additional frames on each side of an clip
ACTION_PADDING = 10

def main(src, num_hits, sk, write_clips):
    reader = VideoReader(src, sk_read=sk)

    if sk:
        logger.info('Initialising reader to use skvideo vreader')
    else:
        logger.info('Initializing reader to use cv2 VideoCapture')

    ball_candidates = None

    if reader.obj_exists('ball_candidates'):
        ball_candidates = reader.load_obj('ball_candidates')
    else:
        # ball_candidates has not already been generated to read and process video
        objs = []
        ball_candidates = []
        for (count, frame, fgmask) in reader.read():
            _, contours, _ = cv2.findContours(fgmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # Find new contours and pair them with old contours
            new_objs = select_contours(frame, contours, 25, count)
            objs = pair_contours(objs, new_objs, count)
            for idx, obj in reversed(list(enumerate(objs))): # Iterates backwards b/c removals
                obj.draw_contour(frame)
                # Remove untrackable (dead) contour tracks and make them a ball candidate
                if not obj.is_trackable():
                    objs.pop(idx)
                    if obj.get_lifetime() > MIN_BALL_LIFETIME:
                        ball_candidates.append(obj)
            reader.write_image(frame, 'contours')

        # Check all remaining contours as ball candidates
        for obj in objs:
            objs.pop(idx)
            if obj.get_lifetime() > MIN_BALL_LIFETIME:
                ball_candidates.append(obj)

        bg = reader.get_background()
        reader.write_image(bg, 'bg')

        reader.dump_obj(ball_candidates, 'ball_candidates')

    # Evaluate all ball candidates and replace then with a 2 tuple (confidence, contourtrack)
    for idx in range(len(ball_candidates)):
        obj = ball_candidates[idx]
        ball_candidates[idx] = (obj.is_ball(), obj)

    # Split and concatenate ball candidates to clean them up
    # TODO try flipping order once ContourTrack concatened_with interpolates
    split_ball_candidates(ball_candidates)
    #concatenate_ball_candidates(ball_candidates)

    # Show ball candidate calculations in order for debugging
    best_balls = sorted(ball_candidates, reverse=True)
    logger.debug('\n\nBEST_VBALLS:')
    for idx, ball in enumerate(best_balls):
        logger.info('Calculating vball %d: %s' % (idx, ball))
        ball[1].is_ball()

    # Write images showing ball candidates in order in groups
    CANDIDATE_GROUP_SIZE = 10
    for i in range(len(ball_candidates) / CANDIDATE_GROUP_SIZE + 1):
        bounds = (i * CANDIDATE_GROUP_SIZE, min((i + 1) * CANDIDATE_GROUP_SIZE, len(ball_candidates)))
        positions = []
        bg = reader.read_image('bg')
        for idx in range(bounds[0], bounds[1]):
            ball = best_balls[idx]
            ball[1].draw_path(bg)
            ball[1].draw_contour(bg)
            positions.append((idx, ball[1].pos_list[-1]))
            
        for pos in positions:
            cv2.putText(bg, str(pos[0]), pos[1], cv2.FONT_HERSHEY_SIMPLEX, 1, HSV_COLOR['WHITE'], 3)

        reader.write_image(bg, 'best_balls%d-%d' % bounds)

    # Find bounds for clipping video
    best_balls = best_balls[:num_hits]
    bounds = []
    for ball in best_balls:
        bounds.append((ball[1].get_birth_frame() - ACTION_PADDING, ball[1].get_last_seen_frame() + ACTION_PADDING))
    logger.info(bounds)

    if not write_clips:
        return

    # Clip video and write clips to correct output folder
    video_outputs = [0] * num_hits
    for i in range(num_hits):
        video_outputs[i] = skvideo.io.FFmpegWriter('out/' + src + '/' + src + '-' + str(i) + '.mp4')
    for (count, frame, fgmask) in reader.read(get_bg=False):
        for i in range(num_hits):
            if count >= bounds[i][0] and count <= bounds[i][1]:
                video_outputs[i].writeFrame(cv2.cvtColor(frame, cv2.COLOR_HSV2RGB))

    for i in range(num_hits):
        logger.info('Writing ' + src + '-' + str(i) + '.mp4')
        video_outputs[i].close()
    logger.info('Done')

def usage():
    print 'Usage: %s file_name num_hits ' % (re.sub('^.*/','',sys.argv[0]))
    print '         file_name: The name of the video in the res folder without .mp4 extension'
    print '         num_hits: The number of hits to clip from the video'
    print '         -s: Use the skvideo vreader (default is opencv VideoCapture)'
    print '         -o: Do not write out the clipped videos'
    print ' '
    print '    Generate video clips from vball sample'
    print ' '
    print '\n'

if __name__ == '__main__':
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'so')
        if len(args) is not 2:
            usage()
            sys.exit(1)
        main(args[0], int(args[1]), ('-s', '') in opts, ('-o', '') not in opts)
    except getopt.GetoptError:
        usage()
        sys.exit(1)
