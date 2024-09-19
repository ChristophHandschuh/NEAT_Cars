# NEAT Algorithm learns to drive a car
import pygame
import os
import math
import sys
import neat

SCREEN_WIDTH = 1536
SCREEN_HEIGHT = 864
SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

TRACK = pygame.image.load("track_normal.png")

class Car:
    def __init__(self):
        self.x = 675
        self.y = 750
        self.angle = -10
        self.speed = 1  # Keep this positive
        self.width = 40
        self.height = 15
        self.color = (255, 0, 0)  # Red
        self.radar_count = 6
        self.radar_length = 110
        self.radar_angles = [-90, -30, 0, 30, 90]  # Adjusted angles
        self.radar_readings = []
        self.distance_traveled = 0
        self.last_position = (self.x, self.y)
        self.time_alive = 0
        self.total_rotation = 0
        self.last_angle = 0

    def draw_car(self, screen):
        # Draw the car
        rotated_rect = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        pygame.draw.rect(rotated_rect, self.color, (0, 0, self.width, self.height))
        rotated = pygame.transform.rotate(rotated_rect, -self.angle)
        screen.blit(rotated, rotated.get_rect(center=(self.x, self.y)))

    def draw_radar(self, screen):
        self.radar_readings = []  # Reset radar readings
        for radar_angle in self.radar_angles:
            x = self.x-10
            y = self.y
            angle = self.angle + radar_angle

            for i in range(self.radar_length):
                x -= math.cos(math.radians(-angle))
                y += math.sin(math.radians(-angle))

                # Check if the point is within screen boundaries
                if not (0 <= int(x) < SCREEN_WIDTH and 0 <= int(y) < SCREEN_HEIGHT):
                    break

                color = screen.get_at((int(x), int(y)))
                if color == (255, 255, 255, 255):  # White (track)
                    break

            distance = min(math.sqrt((x - self.x+10)**2 + (y - self.y)**2), self.radar_length)
            self.radar_readings.append(distance)
            
            # Draw radar line up to the detected point
            pygame.draw.line(screen, (0, 255, 0), (self.x-10, self.y), (x, y), 1)

    def update(self):
        # Update car position based on speed and angle
        self.x -= math.cos(math.radians(-self.angle)) * self.speed  # Changed to subtraction
        self.y += math.sin(math.radians(-self.angle)) * self.speed  # Changed to addition
        new_position = (self.x, self.y)
        self.distance_traveled += math.dist(self.last_position, new_position)
        self.last_position = new_position
        self.time_alive += 1
        self.total_rotation += abs(self.last_angle) - abs(self.angle)
        self.last_angle = self.angle

    def alive(self, screen):
        return screen.get_at((int(self.x-10), int(self.y))) == (255, 255, 255, 255)

    def get_sensor_data(self):
        return self.radar_readings

    def get_reward(self):
        reward = self.distance_traveled * 0.1
        reward += self.time_alive * 0.01
        reward += self.speed * 0.5
        return reward

    def get_total_distance(self):
        return self.distance_traveled

    def get_total_rotation(self):
        return self.total_rotation

def eval_genomes(genomes, config):
    pygame.init()
    clock = pygame.time.Clock()

    # Create a list to store all cars and their corresponding networks
    cars = []
    nets = []

    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        car = Car()
        car.draw_radar(SCREEN)
        cars.append(car)
        nets.append(net)
        genome.fitness = 0

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        SCREEN.blit(TRACK, (0, 0))

        # Update and draw all cars
        for i, car in enumerate(cars):
            if min(car.get_sensor_data()) > 10 and -1720 < car.get_total_rotation() < 1720:# and car.get_total_distance() < 4000:
                sensor_data = car.get_sensor_data()
                output = nets[i].activate(sensor_data)

                # Use neural network output to control the car
                car.angle += output[0] * 10  # Adjust steering
                car.speed = (output[1] + 1.4) * 10  # Adjust speed

                car.update()
                genomes[i][1].fitness += car.get_reward()

                car.draw_car(SCREEN)
                car.draw_radar(SCREEN)
            else:
                # Remove car and its network if it crashes or over-rotates
                cars.pop(i)
                nets.pop(i)

        # End the generation if all cars have crashed
        if len(cars) == 0:
            running = False

        pygame.display.flip()
        clock.tick(60)

def run_neat(config_path):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    winner = p.run(eval_genomes, 200)  # Run for 50 generations

if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config-feedforward.txt")
    run_neat(config_path)
