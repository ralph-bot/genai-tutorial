import pygame
import random
import sys

# 初始化 Pygame
pygame.init()

# 游戏常量
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
GRID_SIZE = 20
GRID_WIDTH = WINDOW_WIDTH // GRID_SIZE
GRID_HEIGHT = WINDOW_HEIGHT // GRID_SIZE

# 颜色定义
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)

# 方向常量
UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)

class Snake:
    def __init__(self):
        self.length = 1
        self.positions = [((GRID_WIDTH // 2), (GRID_HEIGHT // 2))]
        self.direction = random.choice([UP, DOWN, LEFT, RIGHT])
        self.color = GREEN
        self.score = 0
    
    def get_head_position(self):
        return self.positions[0]
    
    def turn(self, direction):
        # 禁止蛇直接反向移动
        if (direction[0] * -1, direction[1] * -1) == self.direction:
            return
        self.direction = direction
    
    def move(self):
        head_x, head_y = self.get_head_position()
        dir_x, dir_y = self.direction
        new_head = ((head_x + dir_x) % GRID_WIDTH, (head_y + dir_y) % GRID_HEIGHT)
        
        # 检查是否撞到自己
        if len(self.positions) > 2 and new_head in self.positions[2:]:
            self.reset()
        else:
            self.positions.insert(0, new_head)
            if len(self.positions) > self.length:
                self.positions.pop()
    
    def reset(self):
        self.length = 1
        self.positions = [((GRID_WIDTH // 2), (GRID_HEIGHT // 2))]
        self.direction = random.choice([UP, DOWN, LEFT, RIGHT])
        self.score = 0
    
    def render(self, surface):
        for p in self.positions:
            rect = pygame.Rect(
                p[0] * GRID_SIZE,
                p[1] * GRID_SIZE,
                GRID_SIZE,
                GRID_SIZE
            )
            pygame.draw.rect(surface, self.color, rect)
            pygame.draw.rect(surface, BLACK, rect, 1)  # 边框
    
    def check_food_collision(self, food_position):
        return self.get_head_position() == food_position

class Food:
    def __init__(self):
        self.position = (0, 0)
        self.color = RED
        self.randomize_position([])
    
    def randomize_position(self, snake_positions):
        while True:
            new_position = (
                random.randint(0, GRID_WIDTH - 1),
                random.randint(0, GRID_HEIGHT - 1)
            )
            if new_position not in snake_positions:
                self.position = new_position
                break
    
    def render(self, surface):
        rect = pygame.Rect(
            self.position[0] * GRID_SIZE,
            self.position[1] * GRID_SIZE,
            GRID_SIZE,
            GRID_SIZE
        )
        pygame.draw.rect(surface, self.color, rect)
        pygame.draw.rect(surface, BLACK, rect, 1)  # 边框

def main():
    clock = pygame.time.Clock()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("贪吃蛇游戏")
    
    snake = Snake()
    food = Food()
    
    font = pygame.font.Font(None, 36)
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    snake.turn(UP)
                elif event.key == pygame.K_DOWN:
                    snake.turn(DOWN)
                elif event.key == pygame.K_LEFT:
                    snake.turn(LEFT)
                elif event.key == pygame.K_RIGHT:
                    snake.turn(RIGHT)
                elif event.key == pygame.K_ESCAPE:
                    running = False
        
        snake.move()
        
        # 检查是否吃到食物
        if snake.check_food_collision(food.position):
            snake.length += 1
            snake.score += 10
            food.randomize_position(snake.positions)
        
        # 绘制
        screen.fill(BLACK)
        snake.render(screen)
        food.render(screen)
        
        # 显示分数
        score_text = font.render(f"分数: {snake.score}", True, WHITE)
        screen.blit(score_text, (10, 10))
        
        # 显示操作提示
        hint_text = font.render("使用方向键移动, ESC 退出", True, WHITE)
        screen.blit(hint_text, (10, WINDOW_HEIGHT - 30))
        
        pygame.display.flip()
        clock.tick(10)  # 控制游戏速度
    
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
