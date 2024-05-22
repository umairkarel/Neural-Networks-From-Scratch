import pygame
import random

white = (225,225,225)
black = (0,0,0)
green = (0,225,0)
red = (225,0,0)

width = 450
height = 450
screen = pygame.display.set_mode((width,height))
clock = pygame.time.Clock()
FPS = 60

class Perceptron:
    def __init__(self, alpha=0.01):
        self.weights = []
        self.alpha = alpha
        for i in range(2):
            self.weights.append(random.uniform(-1, 1))

	# Activation Function
    def sign(self,n):
        return 1 if n>=0 else -1

    def guess(self, inputs):
        s = sum([inputs[i]*self.weights[i] for i in range(len(inputs))])
        # print(self.weights)
        return self.sign(s)

    def train(self, inputs, target):
        guess = self.guess(inputs)
        error = target - guess

        for i in range(len(self.weights)):
            self.weights[i] += error*inputs[i]*self.alpha

class Point:
    def __init__(self):
        self.x = random.randint(0,width)
        self.y = random.randint(0,height)
        self.color = black
        self.inputs = [self.x,self.y]
        self.label = 1 if self.x > self.y else -1

    def show(self):
        if self.label == 1:
            fill = 1
        else:
            fill = 0
        pygame.draw.circle(screen, self.color, (self.x, self.y), 4, fill)


model = Perceptron()
points = [Point() for _ in range(100)]

def draw():
    pygame.draw.line(screen, black, (0,0), (width,height), 2)

    for point in points:
		# model.train(point.inputs, point.label)

        guess = model.guess(point.inputs)
        if guess == point.label:
            point.color = green
        else:
            point.color = red

        point.show()

	
# def train():
    # for point in points:
    #     model.train(point.inputs, point.label)

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    # if pygame.mouse.get_pressed()[0]:
    # 	train()

    draw()

    pygame.display.flip()
    screen.fill(white)
    clock.tick(FPS)
pygame.quit()