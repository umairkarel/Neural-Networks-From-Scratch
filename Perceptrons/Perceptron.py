import pygame
import random

white = (255,255,255)
black = (0,0,0)
green = (0,225,0)
red = (225,0,0)
blue = (0,0,225)

width = 500
height = 500
screen = pygame.display.set_mode((width,height))
clock = pygame.time.Clock()
FPS = 60

def map(n, start1, stop1, start2, stop2):
    return ((n-start1)/(stop1-start1))*(stop2-start2)+start2

def f(x):
    return 0.3*x

class Perceptron:
    def __init__(self, n, alpha=0.008):
        self.weights = []
        self.alpha = alpha
        for i in range(n):
            self.weights.append(random.uniform(-1, 1))

    # Activation Function
    def sign(self,n):
        return 1 if n>=0 else -1

    def guess(self, inputs):
        s = sum([inputs[i]*self.weights[i] for i in range(len(inputs))])

        return self.sign(s)

    def guessY(self,x):
        w0 = self.weights[0]/self.weights[1]
        w1 = self.weights[2]/self.weights[1]

        return -(w1 + w0*x)

    def train(self, inputs, target):
        guess = self.guess(inputs)
        error = target - guess

        for i in range(len(self.weights)):
            self.weights[i] += error*inputs[i]*self.alpha

class Point:
    def __init__(self,x=0,y=0):
        self.x = x
        self.y = y
        self.bais = 1
        if not (x and y):
            self.x = random.uniform(-1,1)
            self.y = random.uniform(-1,1)

        self.inputs = [self.x,self.y,self.bais]
        self.label = 1 if self.y > f(self.x) else -1

    def pixelX(self):
        return int(map(self.x, -1,1,0,width))

    def pixelY(self):
        return int(map(self.y, -1,1,height,0))

    def show(self):
        if self.label == 1:
            fill = 1
        else:
            fill = 0

        px = self.pixelX()
        py = self.pixelY()

        pygame.draw.circle(screen, black, (px, py), 8, fill)


model = Perceptron(3)
points = [Point() for _ in range(200)]
index = 0

def draw():
    global index

    p1 = Point(-1, f(-1))
    p2 = Point(1, f(1))
    pygame.draw.line(screen, blue, (p1.pixelX(),p1.pixelY()), (p2.pixelX(),p2.pixelY()), 2)

    p3 = Point(-1, model.guessY(-1))
    p4 = Point(1, model.guessY(1))
    pygame.draw.line(screen, black, (p3.pixelX(),p3.pixelY()),(p4.pixelX(),p4.pixelY()), 1)

    # SGD
    point = points[index]
    model.train(point.inputs, point.label)

    for point in points:
        guess = model.guess(point.inputs)
        if guess == point.label:
            color = green
        else:
            color = red

        point.show()
        pygame.draw.circle(screen, color, (point.pixelX(), point.pixelY()), 4, 0)

        # Batch GD
        # model.train(point.inputs, point.label)

    # SGD
    index += 1
    if (index >= len(points)):
        index = 0

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    draw()

    pygame.display.flip()
    screen.fill(white)
    clock.tick(FPS)
pygame.quit()