from PIL import Image

def main():
    im = Image.open("source/terrain.png")
    test = im.getpixel((230, 327))[0:3]
    print(test)
    print(im.getpixel((230, 327))) # black
    print(im.getpixel((239, 307))) # blue
    print(im.size)

def check_terrain(x, y, maxDistance, image, color):
    size = image.size
    for i in range(maxDistance + 1):
        for j in range(maxDistance + 1 - i):
            if y - j >= 0 and x - i >= 0 and get_terrain(x-i, y-j, image) != color:
                return True
            if y - j >= 0 and x + i < size[0] and get_terrain(x+i, y-j, image) != color:
                return True
            if y + j < size[1] and x - i >= 0 and get_terrain(x-i, y+j, image) != color:
                return True
            if y + j < size[1] and x + i < size[0] and get_terrain(x+i, y+j, image) != color:
                return True
    return False


if __name__ == "__main__":
    main()