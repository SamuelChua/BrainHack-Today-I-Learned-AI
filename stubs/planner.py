import sys
# sys.path.insert(0, r'C:\Users\intern\Downloads\til-final-2022.1.1-best\til-final-2022.1.1-lower\pyastar2d\src')
# print(sys.path)
from queue import PriorityQueue
from typing import List, Tuple, TypeVar, Dict
from tilsdk.localization import *
import heapq
#import pyastar2d
import numpy as np
from PIL import Image
from numpy import asarray
import sys
import matplotlib.pyplot as plt
import cv2



class Node():
    """A node class for A* Pathfinding"""

    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position

        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position


def astar(maze, start, end):
    """Returns a list of tuples as a path from the given start to the given end in the given maze"""

    # Create start and end node
    start_node = Node(None, start)
    start_node.g = start_node.h = start_node.f = 0
    end_node = Node(None, end)
    end_node.g = end_node.h = end_node.f = 0

    # Initialize both open and closed list
    open_list = []
    closed_list = []

    # Add the start node
    open_list.append(start_node)

    # Loop until you find the end
    while len(open_list) > 0:

        # Get the current node
        current_node = open_list[0]
        current_index = 0
        for index, item in enumerate(open_list):
            if item.f < current_node.f:
                current_node = item
                current_index = index

        # Pop current off open list, add to closed list
        open_list.pop(current_index)
        closed_list.append(current_node)

        # Found the goal
        if current_node == end_node:
            path = []
            current = current_node
            while current is not None:
                path.append(current.position)
                current = current.parent
            return path[::-1] # Return reversed path

        # Generate children
        children = []
        for new_position in [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]: # Adjacent squares

            # Get node position
            node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])

            # Make sure within range
            if node_position[0] > (len(maze) - 1) or node_position[0] < 0 or node_position[1] > (len(maze[len(maze)-1]) -1) or node_position[1] < 0:
                continue

            # Make sure walkable terrain
            if maze[node_position[0]][node_position[1]] != 0:
                continue

            # Create new node
            new_node = Node(current_node, node_position)

            # Append
            children.append(new_node)

        # Loop through children
        for child in children:

            # Child is on the closed list
            for closed_child in closed_list:
                if child == closed_child:
                    continue

            # Create the f, g, and h values
            child.g = current_node.g + 1
            child.h = ((child.position[0] - end_node.position[0]) ** 2) + ((child.position[1] - end_node.position[1]) ** 2)
            child.f = child.g + child.h

            # Child is already in the open list
            for open_node in open_list:
                if child == open_node and child.g > open_node.g:
                    continue

            # Add the child to the open list
            open_list.append(child)
T = TypeVar('T')

class NoPathFoundException(Exception):
    pass


class PriorityQueue:
    def __init__(self):
        self.elements: List[Tuple[float, T]] = []

    def is_empty(self) -> bool:
        return not self.elements

    def put(self, item: T, priority: float):
        heapq.heappush(self.elements, (priority, item))

    def get(self) -> T:
        return heapq.heappop(self.elements)[1]


class Planner:
    def __init__(self, map_:SignedDistanceGrid=None, sdf_weight:float=0.0):
        '''
        Parameters
        ----------
        map : SignedDistanceGrid
            Distance grid map
        sdf_weight: float
            Relative weight of distance in cost function.
        '''
        self.map = map_
        self.sdf_weight = sdf_weight

    def update_map(self, map:SignedDistanceGrid):
        '''Update planner with new map.'''
        self.map = map

    def heuristic(self, a:GridLocation, b:GridLocation) -> float:
        '''Planning heuristic function.
        
        Parameters
        ----------
        a: GridLocation
            Starting location.
        b: GridLocation
            Goal location.
        '''
        return euclidean_distance(a, b)

    def plan(self, start:RealLocation, goal:RealLocation) -> List[RealLocation]:
        '''Plan in real coordinates.
        
        Raises NoPathFileException path is not found.

        Parameters
        ----------
        start: RealLocation
            Starting location.
        goal: RealLocation
            Goal location.
        
        Returns
        -------
        path
            List of RealLocation from start to goal.
        '''

    

        path = self.plan_grid(self.map.real_to_grid(start), self.map.real_to_grid(goal))
        return [self.map.grid_to_real(wp) for wp in path]

    def gradient(self,Listofcoordinates):
    #numberofcoordinates=len(Listofcoordinates)
        output =[]
        for i in range(1,len(Listofcoordinates)-1):
            x0 = Listofcoordinates[i-1][0]
            y0 = Listofcoordinates[i-1][1]
            x1 = Listofcoordinates[i][0]
            y1 = Listofcoordinates[i][1]
            x2 = Listofcoordinates[i+1][0]
            y2 = Listofcoordinates[i+1][1]
            # if y2 == y1 = y0 or x2=x1=x0:
            if x0 == x1 and y1== y2:
                output.append([x1, y1])

            elif y0 == y1 and x1== x2:
                output.append([x1, y1])

            elif not(x0 == x1 and x1== x2) and not(y0 == y1 and y1 == y2):
                if abs((y2 - y1) / (x2 - x1) - (y1 - y0) / (x1 - x0)) < 0.1:
                    pass
                else:
                    output.append([x1, y1])
        output.append((Listofcoordinates[i+1][0],Listofcoordinates[i+1][1]))
        return output

    def plan_grid(self, start:GridLocation, goal:GridLocation) -> List[GridLocation]:
        '''Plan in grid coordinates.
        
        Raises NoPathFileException path is not found.

        Parameters
        ----------
        start: GridLocation
            Starting location.
        goal: GridLocation
            Goal location.
        
        Returns
        -------
        path
            List of GridLocation from start to goal.
        '''


        # img = cv2.imread("/home/dh/Downloads/test_5cm.png")
        # img = img[:,:,1]
        # kernel = np.ones((5,5),np.float32)
        # img = np.float32(img)
        # img[img == 0] = 1
        # img[img == 255 ] = float('inf')

        # The start and goal coordinates are in matrix coordinates (i, j).
        start = (139-start[0],start[1])
        goal = (139-goal[0],goal[1])
        # The minimum cost must be 1 for the heuristic to be valid.
        print(start,goal)
        weights = self.map.grid
        print('hi',weights.shape)

        print("Cost matrix:")
        # print(weights)
        import sys
        import numpy
        numpy.set_printoptions(threshold=sys.maxsize)
        # print(np.max(self.map.grid))
        weights[weights > 0 ] = 20
        weights[weights <= 0 ] = 1
        weights[weights ==20 ] = 0

        weights = weights.tolist()


        # weights[weights <= 0 ] = float(1000000000)
        # weights =  weights - maximum 
        # weights = np.absolute(weights) + 1
        # weights = weights.T
        # c1 = real_to_grid_exact(start,20)
        # c2 = real_to_grid_exact(goal,20)
        # c1 = grid_to_real(start,0.05)
        # c2 = grid_to_real(goal,0.05)
        # print(c1,c2)
        
        # start= (int(c1[0]),int(c1[1]))
        # goal= (int(c2[0]),int(c2[1]))
        # weights = [ [0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0],
        #             [0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0],
        #             [0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0],
        #             [0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0],
        #             [0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0],
        #             [0,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,0,1,1,1,1],
        #             [0,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,0,1,1,1,1],
        #             [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        #             [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        #             [0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,0,1,1,1,1],
        #             [0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,0,1,1,1,1],
        #             [0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0],
        #             [0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0],
        #             [0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0],
        #             [0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0]]
                
        # path = pyastar2d.astar_path(weights,start, goal, allow_diagonal=True)
        path= astar(weights, start, goal)
        path = np.array(path)
        print(path)

        print (path)
        path = self.gradient(path)
        path = np.array(path)
        l1 = []

        print(l1)
        print(path)
        for i in range(path.shape[0]):
            l1.append((path[i,0],path[i,1]))
        
        print('path:',path)

        # The path is returned as a numpy array of (i, j) coordinates.
        print(f"Shortest path from {start} to {goal} found:")
        print(l1)
        #fig, ax = plt.subplots()
        # plt.imshow(img_map)
        
        # x, y = path.T
        # plt.scatter(y,x)
        # plt.show()

        
        # fig, ax = plt.subplots()
        # plt.imshow(img_map)
        # plt.imshow(weights)
        # x, y = path.T
        # plt.scatter(y,x)
        # plt.show()

        # if not self.map:
        #     raise RuntimeError('Planner map is not initialized.')

        # frontier = PriorityQueue()
        # frontier.put(start, 0)
        # for i, coord in enumerate(path):
        #     frontier.put(coord, i+1)
        # came_from: Dict[GridLocation, GridLocation] = {}
        # cost_so_far: Dict[GridLocation, float] = {}
        # came_from[start] = None
        # cost_so_far[start] = 0

        # while not frontier.is_empty():
        #     frontier.get()
        #     # TODO: Participant to complete.
            
        return l1
            # break
        
        if goal not in came_from:
            raise NoPathFoundException

        return self.reconstruct_path(came_from, start, goal)

    def reconstruct_path(self,
                         came_from:Dict[GridLocation, GridLocation],
                         start:GridLocation, goal:GridLocation) -> List[GridLocation]:
        '''Traces traversed locations to reconstruct path.
        
        Parameters
        ----------
        came_from: dict
            Dictionary mapping location to location the planner came from.
        start: GridLocation
            Start location for path.
        goal: GridLocation
            Goal location for path.

        Returns
        -------
        path
            List of GridLocation from start to goal.
        '''
        
        current: GridLocation = goal
        path: List[GridLocation] = []
        
        while current != start:
            path.append(current)
            current = came_from[current]
            
        # path.append(start)
        path.reverse()
        return path




