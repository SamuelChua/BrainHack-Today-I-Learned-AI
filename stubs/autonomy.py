import sys
import numpy

root_dir = r'C:\\Users\\intern\\Downloads\\til-final-2022.1.1-best\\til-final-2022.1.1-lower'
import time

import logging
from typing import List

from tilsdk import *                                            # import the SDK
from tilsdk.utilities import PIDController, SimpleMovingAverage # import optional useful things
from tilsdk.mock_robomaster.robot import Robot                  # Use this for the simulator
#from robomaster.robot import Robot                             # Use this for real robot

# Import your code
from cv_service import CVService, MockCVService
from nlp_service import NLPService, MockNLPService
from planner import Planner

# Setup logging in a nice readable format
logging.basicConfig(level=logging.INFO,
                    format='[%(levelname)5s][%(asctime)s][%(name)s]: %(message)s',
                    datefmt='%H:%M:%S')

# Define config variables in an easily accessible location
# You may consider using a config file
REACHED_THRESHOLD_M = 0.3   # TODO: Participant may tune.
ANGLE_THRESHOLD_DEG = 20.0  # TODO: Participant may tune.
ROBOT_RADIUS_M = 0.17    # TODO: Participant may tune.
NLP_MODEL_DIR = r'C:\\Users\\intern\\Downloads\\til-final-2022.1.1-best\\til-final-2022.1.1-lower\\stubs\\model\\model_checkpoints4350+2epoch_huge2_smaller10+11-20220618T165552Z-001'     # TODO: Participant to fill in.
CV_MODEL_DIR = f'{root_dir}\\stubs\\model\\model_final_66.4%_15k_start_fresh_egg.pth'           # TODO: Participant to fill in.

def gen(t):
    if len(t) == 1:
        yield from range(t[0])

    else:  
        for e in range(t[0]):
            for g in gen(t[1:]):
                yield [e, *([g] if isinstance(g, int) else g)]

def get_coordinates(dim):
    return list(gen(dim))


# NLP_MODEL_DIR = f'{root_dir}\\stubs\\model\\model_checkpoints4350+2epoch_huge2_smaller10+11'   # TODO: Participant to fill in.
# CV_MODEL_DIR = f'{root_dir}\\stubs\\model\\model_final_66.4%_15k_start_fresh_egg.pth'            # TODO: Participant to fill in.



# Convenience function to update locations of interest.
def update_locations(old:List[RealLocation], new:List[RealLocation]) -> None:
    '''Update locations with no duplicates.'''
    if new:
        for loc in new:
            if loc not in old:
                logging.getLogger('update_locations').info('New location of interest: {}'.format(loc))
                old.append(loc)


def main():
    # Initialize services
    cv_service = CVService(model_dir=CV_MODEL_DIR)
    #cv_service = MockCVService(model_dir=CV_MODEL_DIR)
    #nlp_service = MockNLPService(model_dir=NLP_MODEL_DIR)
    nlp_service = NLPService(model_dir=NLP_MODEL_DIR)
    loc_service = LocalizationService(host='localhost', port=5566)
    rep_service = ReportingService(host='localhost', port=5566)
    robot = Robot()
    robot.initialize(conn_type="sta")
    robot.camera.start_video_stream(display=False, resolution='720p')


    rep_service.start_run()

    # Initialize planner
    map_:SignedDistanceGrid = loc_service.get_map()
    map_ = map_.dilated(1.5*ROBOT_RADIUS_M/map_.scale)
    planner = Planner(map_, sdf_weight=0.5)

    # Initialize variables
    seen_clues = set()
    curr_loi:RealLocation = None
    path:List[RealLocation] = []
    lois:List[RealLocation] = []
    curr_wp:RealLocation = None
    past_lois:List[RealLocation] = []

    grid_radius = int(ROBOT_RADIUS_M/5 * planner.map.grid.shape[1])
    padding = np.array(get_coordinates((grid_radius*2+1,grid_radius*2+1)))
    padding = padding - grid_radius

    # Initialize tracker
    # TODO: Participant to tune PID controller values.
    tracker = PIDController(Kp=(0.55, 0.3), Kd=(0.4, 0.3), Ki=(0.05, 0.0))

    # Initialize pose filter
    pose_filter = SimpleMovingAverage(n=5)

    # Define filter function to exclude clues seen before   
    new_clues = lambda c: c.clue_id not in seen_clues

    # Main loop
    while True:
        # Get new data
        pose, clues = loc_service.get_pose()
        pose = pose_filter.update(pose)
        img = robot.camera.read_cv2_image(strategy='newest')
        
        if not pose:
            # now new data, continue to next iteration.
            continue

        # Filter out clues that were seen before
        clues = list(filter(new_clues, clues))

        # Process clues using NLP and determine any new locations of interest
        if clues:
            new_lois = nlp_service.locations_from_clues(clues)
            update_locations(lois, new_lois)
            seen_clues.update([c.clue_id for c in clues])

        # Process image and detect targets
        targets = cv_service.targets_from_image(img)

        # Submit targets
        if targets:
            logging.getLogger('Main').info('{} targets detected.'.format(len(targets)))
            logging.getLogger('Reporting').info(rep_service.report(pose, img, targets))
        
        if not curr_loi:
            if len(lois) == 0:
                logging.getLogger('Main').info('No more locations of interest.')
                # TODO: You ran out of LOIs. You could perform and random search for new
                # clues or targets 
                
                weights = np.float32(planner.map.grid)
                print('hii',weights.shape)

                print("Cost matrix:")
                
                
                numpy.set_printoptions(threshold=sys.maxsize)
                print(np.max(planner.map.grid))
                maximum = np.max(planner.map.grid)
                
                weights[weights <= 0 ] = float('inf')
                weights =  weights - maximum 
                weights = np.absolute(weights) + 1
                weights = weights.T
                path = None
                #In case the clue is nearby, go abit front and go round 
                # if weights nearby is not inf:
                # robot.chassis.drive_speed(x=0.3, y=0.0, z=0.0)
                # robot.chassis.drive_speed(x=-0.3, y=0.3, z=0.0)
                # robot.chassis.drive_speed(x=-0.3, y=-0.3, z=0.0)
                # robot.chassis.drive_speed(x=0.3, y=-0.3, z=0.0)
                # robot.chassis.drive_speed(x=-0.3, y=0.3, z=0.0)    

                if clues:
                    new_lois = nlp_service.locations_from_clues(clues)
                    update_locations(lois, new_lois)
                    seen_clues.update([c.clue_id for c in clues])
                    break

                #Find other possible paths to go
                while path is None:
                    try:
                        save = np.argwhere(weights < 10)
                        
                        #we chose one index randomly
                        i = np.random.randint(save[0])
                        random_pos = i
                        # print (random_pos)
                        path = planner.plan(pose[:2], random_pos)
                    except:
                        pass
                path.reverse() # reverse so closest wp is last so that pop() is cheap
            
                # for i in random_pos:
                

                # random_pos_1 = random_pos + ROBOT_RADIUS_M (planner.map.grid.shape) #map scale
                # random_pos_2 = random_pos + ROBOT_RADIUS_M
                # random_pos_3 = random_pos + ROBOT_RADIUS_M
                # random_pos_4 = random_pos + ROBOT_RADIUS_M
                
                
                x_coords = padding[:,0]+ random_pos[0]/map_.scale - grid_radius
                y_coords = padding[:,1]+ random_pos[1]/map_.scale - grid_radius
                blacklist_coords = list(map(tuple,np.vstack((x_coords,y_coords)).T))
                
                tuple_random_pos = tuple(random_pos)
                lois.append(tuple_random_pos)
                past_lois+= blacklist_coords
                print (past_lois)
                print (len(past_lois))
                #past_lois.append()
                curr_wp = None
                logging.getLogger('Main').info('Random Path planned.')
                
                
                # Process clues using NLP and determine any new locations of interest
                if clues:
                    new_lois = nlp_service.locations_from_clues(clues)
                    update_locations(lois, new_lois)
                    seen_clues.update([c.clue_id for c in clues])
                    break
            else:
                # Get new LOI
                lois.sort(key=lambda l: euclidean_distance(l, pose), reverse=True)
                curr_loi = lois.pop()
                logging.getLogger('Main').info('Current LOI set to: {}'.format(curr_loi))

                # Plan a path to the new LOI
                logging.getLogger('Main').info('Planning path to: {}'.format(curr_loi))
                path = planner.plan(pose[:2], curr_loi)
                path.reverse() # reverse so closest wp is last so that pop() is cheap
                past_lois.append(curr_loi)
                #past_lois.append()
                curr_wp = None
                logging.getLogger('Main').info('Path planned.')
        else:
            # There is a current LOI objective.
            # Continue with navigation along current path.
            if path and path[-1] not in past_lois: #TODO ADD ROBOT RADIUS COORDS
                # Get next waypoint
                if not curr_wp:
                    curr_wp = path.pop()
                    logging.getLogger('Navigation').info('New waypoint: {}'.format(curr_wp))
                
                # Calculate distance and heading to waypoint
                dist_to_wp = euclidean_distance(pose, curr_wp)
                ang_to_wp = np.degrees(np.arctan2(curr_wp[1]-pose[1], curr_wp[0]-pose[0]))
                ang_diff = -(ang_to_wp - pose[2]) # body frame

                # ensure ang_diff is in [-180, 180]
                if ang_diff < -180:
                    ang_diff += 360

                if ang_diff > 180:
                    ang_diff -= 360

                logging.getLogger('Navigation').debug('ang_to_wp: {}, hdg: {}, ang_diff: {}'.format(ang_to_wp, pose[2], ang_diff))

                # Consider waypoint reached if within a threshold distance
                if dist_to_wp < REACHED_THRESHOLD_M:
                    logging.getLogger('Navigation').info('Reached wp: {}'.format(curr_wp))
                    tracker.reset()
                    curr_wp = None
                    continue
                
                # Determine velocity commands given distance and heading to waypoint
                vel_cmd = tracker.update((dist_to_wp, ang_diff))

                # reduce x velocity
                vel_cmd[0] *= np.cos(np.radians(ang_diff))
                
                # If robot is facing the wrong direction, turn to face waypoint first before
                # moving forward.
                if abs(ang_diff) > ANGLE_THRESHOLD_DEG:
                    vel_cmd[0] = 0.0
                
                # Send command to robot
                robot.chassis.drive_speed(x=vel_cmd[0], z=vel_cmd[1])
                
            else:
                logging.getLogger('Navigation').info('End of path.')
                curr_loi = None
                count = 0
                # TODO: Perform search behaviour? Participant to complete.
                
                for i in range(0, 2):
                    img = robot.camera.read_cv2_image(timeout=2, strategy='newest')
                    # Process image and detect targets
                    targets = cv_service.targets_from_image(img)
                    # Submit targets
                    if targets:
                        logging.getLogger('Main').info('{} targets detected.'.format(len(targets)))
                        logging.getLogger('Reporting').info(rep_service.report(pose, img, targets))
                    else:
                        print ("no targets")
                    img_name = "Image_{}".format(count)
                    cv2.imshow(img_name, img)
                    cv2.waitKey(1)
                    count += 1
                time.sleep(3)
                cv2.destroyAllWindows()

                robot.chassis.drive_speed(x=0.0, y=0.0, z=150)
                for i in range(0, 2):
                    img = robot.camera.read_cv2_image(timeout=2, strategy='newest')
                    # Process image and detect targets
                    targets = cv_service.targets_from_image(img)
                    # Submit targets
                    if targets:
                        logging.getLogger('Main').info('{} targets detected.'.format(len(targets)))
                        logging.getLogger('Reporting').info(rep_service.report(pose, img, targets))
                    else:
                        print ("no targets")
                    img_name = "Image_{}".format(count)
                    cv2.imshow(img_name, img)
                    cv2.waitKey(1)
                    count += 1
                time.sleep(3)
                cv2.destroyAllWindows()

                

                robot.chassis.drive_speed(x=0.0, y=0.0, z=150)
                for i in range(0, 2):
                    img = robot.camera.read_cv2_image(timeout=2, strategy='newest')
                    # Process image and detect targets
                    targets = cv_service.targets_from_image(img)
                    # Submit targets
                    if targets:
                        logging.getLogger('Main').info('{} targets detected.'.format(len(targets)))
                        logging.getLogger('Reporting').info(rep_service.report(pose, img, targets))
                    else:
                        print ("no targets")
                    img_name = "Image_{}".format(count)
                    cv2.imshow(img_name, img)
                    cv2.waitKey(1)
                    count += 1
                time.sleep(3)
                cv2.destroyAllWindows()

                

                

                
                continue

    robot.chassis.drive_speed(x=0.0, y=0.0, z=0.0)  # set stop for safety
    logging.getLogger('Main').info('Mission Terminated.')


if __name__ == '__main__':
    main()