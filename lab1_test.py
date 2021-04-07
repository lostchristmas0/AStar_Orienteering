import math
import sys
from PIL import Image

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
    def __init__(self):
        self.queue = []

    def enqueue(self, vertex):
        if len(self.queue) == 0 or vertex.function[2] >= self.queue[-1].function[2]:
            self.queue.append(vertex)
        else:
            for i in range(len(self.queue)):
                if vertex.function[2] < self.queue[i].function[2]:
                    self.queue.insert(i, vertex)
                    break

    def dequeue(self):
        return self.queue.pop(0)

    def remove(self, vertex):
        self.queue.remove(vertex)

    def isEmpty(self):
        return len(self.queue) == 0

    # def sortAdd(self, neighbor):


class Vertex:
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
        Connect two words by adding the second vertex into the first vertex's neighbor
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
        if v in self.vertList:
            for vx in self.vertList[v].neighbor:
                vx.neighbor.pop(v)
            self.vertList.pop(v)
        # else:
        #     print("didn't find:" + str(v))

    def clearPred(self):
        for v in self.vertList.values():
            v.pred = None


"""
In usual case, slope <= 5, speed = max speed; slope >= 45, speed = 0;
in between is considered as a linear decrease
"""
def get_speed(terrain, slope):
    if slope < 0 or slope >= 90:
        raise TypeError("invalid slope")
    elif 0 <= slope <= 5:
        return TERRAIN_SPEED[terrain]
    elif slope >= 45:
        return 0
    else:
        return (9 / 8 - (1 / 40) * slope) * TERRAIN_SPEED[terrain]


def change_fall(image):
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
    newTerrain = (100, 200, 255)
    newImage = image.copy()
    size = image.size
    for y in range(size[1]):
        for x in range(size[0]):
            if newImage.getpixel((x, y))[0: 3] == (0, 0, 255) and winter_terrain(x, y, 7, image, (0,0,255)):
                newImage.putpixel((x, y), newTerrain)
    return newImage


def change_spring(image, data):
    newTerrain = (0,0,100)
    newImage = image.copy()
    size = image.size
    for y in range(size[1]):
        for x in range(size[0]):
            if newImage.getpixel((x, y))[0: 3] == (0, 0, 255):
                near = spring_terrain(x, y, 15, image, (0,0,255))
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
                if get_terrain(v.coordinate[0], v.coordinate[1], image) == color and not v in checked:
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
    # 3d coordinate
    return math.sqrt(pow((start[0] - destination[0]) * LONGITUDE, 2) + pow((start[1] - destination[1]) * LATITUDE, 2) + pow((start[2] - destination[2]), 2))


def get_slope(start, destination):
    tan = abs(start[2] - destination[2]) / math.sqrt(pow((start[0] - destination[0]) * LONGITUDE, 2) + pow((start[1] - destination[1]) * LATITUDE, 2))
    return math.atan(tan) * 180 / math.pi


def heuristics_time(x0, y0, x1, y1, data):
    # slope = get_slope((x0, y0, data[y0][x0]), (x1, y1, data[y1][x1]))
    time = get_distance((x0, y0, data[y0][x0]), (x1, y1, data[y1][x1])) / get_speed((71, 51, 3), 0)
    return time


def heuristics_speed(x0, y0, x1, y1, image):
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

    # match each coordinate to its terrain and calculate the average speed in this heuristics path
    # average = {}
    # for xy in collection:
    #     color = image.getpixel(xy)[0:3]
    #     if not color in average:
    #         average[color] = 1
    #     else:
    #         average[color] += 1
    # weight = 0
    # total = 0
    # for key in average:
    #     weight += average[key]
    #     total += average[key] * TERRAIN_SPEED[key]
    # return total / weight


def get_terrain(x, y, image):
    color = image.getpixel((x, y))[0:3]
    return color


def create_path(fileName):
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
    # data come from input_elevation(fileName), contains the 3d-coordinate list (x,y,z)
    g = Graph()
    for y in range(len(data)):
        for x in range(len(data[y])):
            coordinate = (x, y, data[y][x])
            terrain = get_terrain(x, y, image)
            if TERRAIN_SPEED[terrain] != 0:
                # print("adding this:", str(coordinate), "color:", str(terrain))
                g.addVertex(coordinate)
                if x != 0:
                    left = (x - 1, y, data[y][x-1])
                    if left in g.vertList and get_speed(terrain, get_slope(coordinate, left)) != 0:
                        g.connect(coordinate, left, get_distance(coordinate, left) / get_speed(terrain, get_slope(coordinate, left)))
                        # print("adding neighbor:", str(left))
                if y != 0:
                    up = (x, y - 1, data[y-1][x])
                    if up in g.vertList and get_speed(terrain, get_slope(coordinate, up)) != 0:
                        g.connect(coordinate, up, get_distance(coordinate, up) / get_speed(terrain, get_slope(coordinate, up)))
                        # print("adding neighbor:", str(up))
                # if x != len(data[y])-1:
                #     right = (x + 1, y, data[y][x+1])
                #     if right in g.vertList and get_speed(terrain, get_slope(coordinate, right)) != 0:
                #         g.connect(coordinate, right, get_distance(coordinate, right) / get_speed(terrain, get_slope(coordinate, right)))
                #         # print("adding neighbor:", str(right))
                # if y != len(data)-1:
                #     down = (x, y + 1, data[y+1][x])
                #     if down in g.vertList and get_speed(terrain, get_slope(coordinate, down)) != 0:
                #         g.connect(coordinate, down, get_distance(coordinate, down) / get_speed(terrain, get_slope(coordinate, down)))
                #         # print("adding neighbor:", str(down))
        # print("finish line", y)
    return g


def aStarSearch(image, data, graph, start, destination):
    # start, destination are 2d

    startCoor = (start[0], start[1], data[start[1]][start[0]])
    destCoor = (destination[0], destination[1], data[destination[1]][destination[0]])
    startVertex = graph.getVertex(startCoor)

    startVertex.modifyH(heuristics_time(start[0], start[1], destination[0], destination[1], data))
    startVertex.modifyF(startVertex.function[0] + startVertex.function[1])
    # if heuristics_speed(start[0], start[1], destination[0], destination[1], image) != 0:
    #     startVertex.modifyH(get_distance(startCoor, destCoor) / heuristics_speed(start[0], start[1], destination[0], destination[1], image))
    #     startVertex.modifyF(startVertex.function[0] + startVertex.function[1])

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
                #     n.modifyH(get_distance(n.coordinate, destCoor) / heuristics_speed(n.coordinate[0], n.coordinate[1], destCoor[0], destCoor[1], image))
                #     n.modifyF(n.function[0] + n.function[1])

                n.pred = current
                visited[n] = n.function[2] # try this!
                search.enqueue(n)
            elif current.function[1] + current.neighbor[n] + n.function[0] < n.function[2]:
                # search.remove(n)
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
    g = buildGraph(data, image)

    # check for bfs function:
    # image = bfsAll(g, 7, image, (100, 200, 255), (0, 0, 255))
    # for v in checked:
    #     print(v.coordinate)

    for i in range(len(path) - 1):
        searchResult = []
        startx = path[i][0]
        starty = path[i][1]
        destx = path[i+1][0]
        desty = path[i+1][1]
        currentPath = [(startx, starty), (destx, desty)]
        aStarSearch(image, data, g, currentPath[0], currentPath[1])
        current = g.getVertex((destx, desty, data[desty][destx]))
        while current != None:
            searchResult.insert(0, current.coordinate)
            current = current.pred
        for c in searchResult:
            totalRoute.append(c[0:2])
        g.clearPred()

    imageCopy = image.copy()
    for p in totalRoute:
        imageCopy.putpixel(p, (243, 9, 9))
    imageCopy.save(outputFile)

    # data = input_elevation("source/mpp.txt")
    # image = Image.open("source/terrain.png")
    # g = buildGraph(data, image)
    #
    # aStarSearch(image, data, g, (230, 327), (276, 279))
    # searchResult = []
    # current = g.getVertex((276, 279, data[279][276]))
    # while current != None:
    #     searchResult.insert(0, current.coordinate)
    #     current = current.pred
    # print(searchResult)
    #
    # imageCopy = image.copy()
    # for p in searchResult:
    #     imageCopy.putpixel(p[0:2], (243, 9, 9))
    # imageCopy.save("source/result_test.png")

    image.close()
    imageCopy.close()

    # gg = Graph()
    # v1 = Vertex((1,1,10))
    # v2 = Vertex((2,1,10))
    # v3 = Vertex((1,2,10))
    # gg.addVertex((1,1,10))
    # gg.connect((1,1,10), (2,1,10), 100)
    # print(gg.getVertex((1,1,10)))



if __name__ == "__main__":
    main()
