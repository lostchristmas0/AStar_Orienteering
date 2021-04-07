"""
file: lab1.py
language: python 3
description: This program takes a map image, an elevation data text file,
and a designed route text file to perform a best route search via A* Search Algorithm.
The best route will be shown in the output image as a red color.
author: Chenghui Zhu, cz3348@rit.edu
"""

import math
import sys
from PIL import Image

"""
The RGB color and the pre-defined speed at different terrains (m/s)
Footpath in fall is re-drawn with (150, 150, 150)
Lake/Swamp/Marsh in winter (walkable) is re-drawn with (100, 200, 255)
Mud area near Lake/Swamp/Marsh in spring (unwalkable) is re-drawn with (0, 0, 100)
"""
OL = 2
RM = 1.5
EMF = 1.8
SRF = 1.4
WF = 1.2
IV = 0
LSM = 0
LSM_W = 1
PR = 2.5
FP = 2.2
FP_S = EMF
TERRAIN_SPEED = {(248, 148, 18): OL, (255, 192, 0): RM,
                 (255, 255, 255): EMF, (2, 208, 60): SRF,
                 (2, 136, 40): WF, (5, 73, 24): IV,
                 (0, 0, 255): LSM, (71, 51, 3): PR,
                 (0, 0, 0): FP, (205, 0, 101): 0,
                 (150, 150, 150): FP_S, (100, 200, 255): LSM_W,
                 (0, 0, 100): LSM}
LONGITUDE = 10.29 # X
LATITUDE = 7.55 # Y


class PriorityQueue:
    """
    The class used as a priority queue, the priority is determined by the
    f(n) function of each vertex
    """
    def __init__(self):
        self.queue = []

    def enqueue(self, vertex):
        """
        The vertex with a lower f(n) will be enqueued in the front of the list
        :param vertex: the vertex to be enqueued
        :return: None
        """
        if len(self.queue) == 0 or vertex.function[2] >= self.queue[-1].function[2]:
            self.queue.append(vertex)
        else:
            for i in range(len(self.queue)):
                if vertex.function[2] < self.queue[i].function[2]:
                    self.queue.insert(i, vertex)
                    break

    def dequeue(self):
        """
        The first vertex in the list will be dequeued
        :return: the first vertex
        """
        return self.queue.pop(0)

    def remove(self, vertex):
        """
        Remove a certain vertex in the list
        :param vertex: the removed vertex
        :return: None
        """
        self.queue.remove(vertex)

    def isEmpty(self):
        return len(self.queue) == 0


class Vertex:
    """
    The vertex class represents a certain 3d-coordinate in the given map and the elevation data.
    coordinate: a 3d-tuple of the certain point in space
    neighbor: the neighbor vertex as the key and the cost time to this neighbor as the value
    pred: the predecessor vertex in the graph
    function: a list of three function used in A* search: h(n), g(n), f(n)
    """
    def __init__(self, coordinate, pred=None):
        self.coordinate = coordinate
        self.neighbor = {}
        self.pred = pred
        self.function = [sys.maxsize, 0, sys.maxsize] # representing h(n), g(n), f(n)

    def addNeighbor(self, vertex, value):
        self.neighbor[vertex] = value

    def modifyH(self, value):
        self.function[0] = value

    def modifyG(self, value):
        self.function[1] = value

    def modifyF(self, value):
        self.function[2] = value

    def __str__(self):
        s = str(self.coordinate) + " neighbor: "
        for n in self.neighbor:
            s += str(n.coordinate) + ", "
        return s


class Graph:
    """
    The graph contains all connected (reachable) vertices in the map.
    vertList: 3d-coordinate as the key, vertex as the value
    """
    def __init__(self):
        self.vertList = {}

    def clear(self):
        self.vertList.clear()

    def addVertex(self, coordinate):
        newVertex = Vertex(coordinate)
        self.vertList[coordinate] = newVertex
        return newVertex

    def getVertex(self, coordinate):
        if coordinate in self.vertList:
            return self.vertList[coordinate]
        else:
            return None

    def connect(self, v1, v2, value):
        """
        Connect two vertices by adding them into each other's neighbor
        :param v1: the first vertex
        :param v2: the second vertex
        :return: None
        """
        if v1 not in self.vertList:
            nv = self.addVertex(v1)
        if v2 not in self.vertList:
            nv = self.addVertex(v2)
        self.vertList[v1].addNeighbor(self.vertList[v2], value)
        self.vertList[v2].addNeighbor(self.vertList[v1], value)

    def disconnect(self, v):
        """
        Disconnect a certain vertex with all its neighbor
        :param v: the vertex to be disconnected
        :return: None
        """
        if v in self.vertList:
            for vx in self.vertList[v].neighbor:
                vx.neighbor.pop(v)
            self.vertList.pop(v)
        # else:
        #     print("didn't find:" + str(v))

    def clearPred(self):
        """
        Clear all vertices' predecessor
        :return: None
        """
        for v in self.vertList.values():
            v.pred = None


def get_speed(terrain, slope):
    """
    Define the speed at different terrain and slope.
    In usual case, slope <= 5, speed = max speed; slope >= 45, speed = 0;
    in between is considered as a linear decrease
    :param terrain: the 3d-tuple as the color of the terrain
    :param slope: the slope in degree
    :return: the modified speed (m/s)
    """
    if slope < 0 or slope >= 90:
        raise TypeError("invalid slope")
    elif 0 <= slope <= 5:
        return TERRAIN_SPEED[terrain]
    elif slope >= 45:
        return 0
    else:
        return (9 / 8 - (1 / 40) * slope) * TERRAIN_SPEED[terrain]


def change_fall(image):
    """
    Change the map based on fall condition. Updated region will be colored with (150, 150, 150)
    :param image: the original image
    :return: the updated image
    """
    newTerrain = (150, 150, 150)
    newImage = image.copy()
    size = image.size
    for y in range(size[1]):
        for x in range(size[0]):
            if newImage.getpixel((x, y))[0: 3] == (0, 0, 0):
                currentNeighbor = set()
                if x > 0:
                    currentNeighbor.add(newImage.getpixel((x - 1, y))[0: 3])
                if x < size[0] - 1:
                    currentNeighbor.add(newImage.getpixel((x + 1, y))[0: 3])
                if y > 0:
                    currentNeighbor.add(newImage.getpixel((x, y - 1))[0: 3])
                if y < size[1] - 1:
                    currentNeighbor.add(newImage.getpixel((x, y + 1))[0: 3])
                if (255, 255, 255) in currentNeighbor:
                    newImage.putpixel((x, y), newTerrain)
    return newImage


def change_winter(image):
    """
    Change the map based on winter condition. Updated region will be colored with (100, 200, 255)
    :param image: the original image
    :return: the updated image
    """
    newTerrain = (100, 200, 255)
    newImage = image.copy()
    size = image.size
    for y in range(size[1]):
        for x in range(size[0]):
            if newImage.getpixel((x, y))[0: 3] == (0, 0, 255) \
                    and winter_terrain(x, y, 7, image, (0, 0, 255)):
                newImage.putpixel((x, y), newTerrain)
    return newImage


def change_spring(image, data):
    """
    Change the map based on spring condition. Updated region will be colored with (0, 0, 100)
    :param image: the original map
    :param data: the original elevation data
    :return: the updated image
    """
    newTerrain = (0, 0, 100)
    newImage = image.copy()
    size = image.size
    for y in range(size[1]):
        for x in range(size[0]):
            if newImage.getpixel((x, y))[0: 3] == (0, 0, 255):
                near = spring_terrain(x, y, 15, image, (0, 0, 255))
                for p in near:
                    if data[p[1]][p[0]] - data[y][x] < 1:
                        newImage.putpixel(p, newTerrain)
    return newImage


def bfs(start, maxDistance, image, color):
    bfsQueue = []
    bfsQueue.append(start)
    checked = set()
    checked.add(start)
    distance = 1
    while distance < maxDistance:
        nextLevel = set()
        while len(bfsQueue) > 0:
            current = bfsQueue.pop(0)
            for v in current.neighbor:
                if get_terrain(v.coordinate[0], v.coordinate[1], image) == color \
                        and not v in checked:
                    nextLevel.add(v)
                    checked.add(v)
        for n in nextLevel:
            bfsQueue.append(n)
        distance += 1
    return checked


def bfsAll(graph, maxDistance, image, color, originalColor):
    edge = set()
    saved = set()
    for v in graph.vertList:
        if get_terrain(v[0], v[1], image) == color:
            vn = graph.getVertex(v).neighbor
            for n in vn:
                if get_terrain(n.coordinate[0], n.coordinate[1], image) != color:
                    edge.add(v)
                    break
    for v in edge:
        saved = saved | bfs(graph.getVertex(v), maxDistance, image, color)
    for v in graph.vertList:
        if get_terrain(v[0], v[1], image) == color and not v in saved:
            image.putpixel((v[0], v[1]), originalColor)
            # graph.disconnect(graph.getVertex(v))
    return image


def winter_terrain(x, y, maxDistance, image, color):
    """
    Check if the specific point required to change color
    :param x: the x-axis in the map
    :param y: the y-axis in the map
    :param maxDistance: depends on given winter condition
    :param image: the original map
    :param color: the updated color
    :return: True if the specific point required to change color
    """
    size = image.size
    for i in range(maxDistance + 1):
        for j in range(maxDistance + 1 - i):
            if y - j >= 0 and x - i >= 0 and get_terrain(x-i, y-j, image) != color:
                return True
            elif y - j >= 0 and x + i < size[0] and get_terrain(x+i, y-j, image) != color:
                return True
            elif y + j < size[1] and x - i >= 0 and get_terrain(x-i, y+j, image) != color:
                return True
            elif y + j < size[1] and x + i < size[0] and get_terrain(x+i, y+j, image) != color:
                return True
    return False


def spring_terrain(x, y, maxDistance, image, color):
    """
    Collect all points required to change color
    :param x: the x-axis in the map
    :param y: the y-axis in the map
    :param maxDistance: depends on given spring condition
    :param image: the original map
    :param color: the updated color
    :return: a set of 2d tuples containing all points that may change color
    """
    size = image.size
    mud = set()
    for i in range(maxDistance + 1):
        for j in range(maxDistance + 1 - i):
            if y - j >= 0 and x - i >= 0 and get_terrain(x-i, y-j, image) != color:
                mud.add((x-i, y-j))
            elif y - j >= 0 and x + i < size[0] and get_terrain(x+i, y-j, image) != color:
                mud.add((x+i, y-j))
            elif y + j < size[1] and x - i >= 0 and get_terrain(x - i, y + j, image) != color:
                mud.add((x-i, y+j))
            elif y + j < size[1] and x + i < size[0] and get_terrain(x + i, y + j, image) != color:
                mud.add((x+i, y+j))
    return mud


def input_elevation(fileName):
    """
    Transfer the elevation info into a list
    :param fileName: the elevation file
    :return: a list of all elevation data
    """
    data = []
    with open(fileName, "r") as file:
        for line in file:
            line = line.strip()
            temp = line.split()
            del temp[-1: -6: -1]
            for i in range(len(temp)):
                temp[i] = float(temp[i])
            data.append(temp)
    return data


def get_distance(start, destination):
    """
    Calculate the distance between two coordinates
    :param start: the start coordinate
    :param destination: the destination coordinate
    :return: the distance
    """
    # 3d coordinate
    return math.sqrt(pow((start[0] - destination[0]) * LONGITUDE, 2)
                     + pow((start[1] - destination[1]) * LATITUDE, 2)
                     + pow((start[2] - destination[2]), 2))


def get_slope(start, destination):
    """
    Calculate the slope between two coodinates
    :param start: the start coordinate
    :param destination: the destination coordinate
    :return: the slope in degree
    """
    tan = abs(start[2] - destination[2]) / math.sqrt(pow((start[0] - destination[0]) * LONGITUDE, 2)
                                                     + pow((start[1] - destination[1]) * LATITUDE, 2))
    return math.atan(tan) * 180 / math.pi


def heuristics_time(x0, y0, x1, y1, data):
    """
    Calculate the heuristics time of a certain coordinate
    :param x0: the x-axis of the coordinate
    :param y0: the y-axis of the coordinate
    :param x1: the x-axis of the destination
    :param y1: the y-axis of the destination
    :param data: the elevation data
    :return: the heuristics time
    """
    time = get_distance((x0, y0, data[y0][x0]), (x1, y1, data[y1][x1])) / max(TERRAIN_SPEED.values())
    return time


def heuristics_speed(x0, y0, x1, y1, image):
    """
    Back-up heuristics speed function: finding the max speed based on crossing terrains
    P.S. Not being used in this program
    :param x0: the x-axis of the coordinate
    :param y0: the y-axis of the coordinate
    :param x1: the x-axis of the destination
    :param y1: the y-axis of the destination
    :param image: the map being used
    :return: the possible max speed
    """
    # return max(TERRAIN_SPEED.values())
    # collect all coordinates in the heuristics path
    collection = []
    dx = x1 - x0
    dy = y1 - y0
    gap = max(abs(dx), abs(dy))
    for i in range(gap):
        inter = (x0 + i*dx/gap, y0 + i*dy/gap)
        temp1 = None
        temp2 = None
        if inter[0] % 1 == 0 and inter[1] % 1 == 0:
            collection.append(inter)
            continue
        elif inter[0] % 1 != 0:
            temp1 = (math.floor(inter[0]), inter[1])
            temp2 = (math.ceil(inter[0]), inter[1])
        elif inter[1] % 1 != 0:
            temp1 = (inter[0], math.floor(inter[1]))
            temp2 = (inter[0], math.ceil(inter[1]))
        collection.append(temp1)
        collection.append(temp2)
    collection.append((x1, y1))
    # match each coordinate to its terrain and find the max speed in this heuristics path
    speedSet = set()
    for xy in collection:
        color = image.getpixel(xy)[0:3]
        speed = TERRAIN_SPEED[color]
        if not speed in speedSet:
            speedSet.add(speed)
    return max(speedSet)


def get_terrain(x, y, image):
    """
    Find the 3d-tuple of the color of a specific 2d-coordinate
    :param x: the x-axis in the image
    :param y: the y-axis in the image
    :param image: the original image
    :return: the 3d-tuple of color
    """
    color = image.getpixel((x, y))[0:3]
    return color


def create_path(fileName):
    """
    Transfer the route info into a list
    :param fileName: the path-file
    :return: a list of all start and destination 2d-points
    """
    path = []
    with open(fileName, "r") as file:
        for line in file:
            line = line.strip()
            temp = line.split()
            for i in range(len(temp)):
                temp[i] = int(temp[i])
            path.append(temp)
    return path


def buildGraph(data, image):
    """
    Building a graph contains all reachable pixels
    :param data: the elevation data
    :param image: the map being used
    :return: the graph
    """
    # data come from input_elevation(fileName), contains the 3d-coordinate list (x,y,z)
    g = Graph()
    for y in range(len(data)):
        for x in range(len(data[y])):
            coordinate = (x, y, data[y][x])
            terrain = get_terrain(x, y, image)
            if TERRAIN_SPEED[terrain] != 0:
                g.addVertex(coordinate)
                if x != 0:
                    left = (x - 1, y, data[y][x-1])
                    if left in g.vertList and get_speed(terrain, get_slope(coordinate, left)) != 0:
                        g.connect(coordinate, left,
                                  get_distance(coordinate, left) /
                                  get_speed(terrain, get_slope(coordinate, left)))
                if y != 0:
                    up = (x, y - 1, data[y-1][x])
                    if up in g.vertList and get_speed(terrain, get_slope(coordinate, up)) != 0:
                        g.connect(coordinate, up,
                                  get_distance(coordinate, up) /
                                  get_speed(terrain, get_slope(coordinate, up)))
        # print("finish line", y)
    return g


def aStarSearch(image, data, graph, start, destination):
    """
    Performing an A* Search from the start to the destination vertex
    :param image: the map being used
    :param data: the elevation data
    :param graph: the graph containing all reachable vertices
    :param start: the 2d-tuple of start pixel's coordinate
    :param destination: the 2d-tuple of destination pixel's coordinate
    :return: None
    """
    startCoor = (start[0], start[1], data[start[1]][start[0]])
    destCoor = (destination[0], destination[1], data[destination[1]][destination[0]])
    startVertex = graph.getVertex(startCoor)

    startVertex.modifyH(heuristics_time(start[0], start[1], destination[0], destination[1], data))
    startVertex.modifyF(startVertex.function[0] + startVertex.function[1])
    # if heuristics_speed(start[0], start[1], destination[0], destination[1], image) != 0:
    #     startVertex.modifyH(get_distance(startCoor, destCoor) /
    #     heuristics_speed(start[0], start[1], destination[0], destination[1], image))
    #     startVertex.modifyF(startVertex.function[0] + startVertex.function[1])

    # visited contains all vertices with their best g(n)
    visited = {}
    search = PriorityQueue()
    search.enqueue(startVertex)
    while not search.isEmpty():
        current = search.dequeue()
        if current.coordinate[0:2] == destination:
            break
        visited[current] = current.function[2]
        for n in current.neighbor:
            if not n in visited:
                n.modifyG(current.function[1] + current.neighbor[n])

                n.modifyH(heuristics_time(n.coordinate[0], n.coordinate[1], destCoor[0], destCoor[1], data))
                n.modifyF(n.function[0] + n.function[1])
                # if heuristics_speed(n.coordinate[0], n.coordinate[1], destCoor[0], destCoor[1], image) != 0:
                #     n.modifyH(get_distance(n.coordinate, destCoor) /
                #     heuristics_speed(n.coordinate[0], n.coordinate[1], destCoor[0], destCoor[1], image))
                #     n.modifyF(n.function[0] + n.function[1])

                n.pred = current
                visited[n] = n.function[2]
                search.enqueue(n)
            elif current.function[1] + current.neighbor[n] + n.function[0] < n.function[2]:
                n.modifyG(current.function[1] + current.neighbor[n])
                n.modifyF(n.function[0] + n.function[1])
                n.pred = current
                visited[n] = n.function[2]
                search.enqueue(n)


def main():
    image = Image.open(sys.argv[1])
    data = input_elevation(sys.argv[2])
    path = create_path(sys.argv[3])
    season = sys.argv[4]
    outputFile = sys.argv[5]
    if season.lower() == "winter":
        image = change_winter(image)
    elif season.lower() == "spring":
        image = change_spring(image, data)
    elif season.lower() == "fall":
        image = change_fall(image)

    totalRoute = []
    totalDistance = 0
    # totalTime = 0
    g = buildGraph(data, image)
    for i in range(len(path) - 1):
        searchResult = []
        startx = path[i][0]
        starty = path[i][1]
        destx = path[i+1][0]
        desty = path[i+1][1]
        currentPath = [(startx, starty), (destx, desty)]
        aStarSearch(image, data, g, currentPath[0], currentPath[1])
        current = g.getVertex((destx, desty, data[desty][destx]))
        # totalTime += current.function[2]
        while current != None:
            temp = current
            searchResult.insert(0, current.coordinate)
            current = current.pred
            if current != None:
                totalDistance += get_distance(temp.coordinate, current.coordinate)
        for c in searchResult:
            totalRoute.append(c[0:2])
        g.clearPred()

    imageCopy = image.copy()
    for p in totalRoute:
        imageCopy.putpixel(p, (243, 9, 9))
    imageCopy.save(outputFile)
    print("Route:", sys.argv[3], "\nSeason:", sys.argv[4], "\nTotal distance:", totalDistance, "m")

    image.close()
    imageCopy.close()


if __name__ == "__main__":
    main()
