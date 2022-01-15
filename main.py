import sys, time, random, pygame
from collections import deque
import cv2 as cv, mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
pygame.init()

# Initialize required elements/environment
VID_CAP = cv.VideoCapture(2)
window_size = (VID_CAP.get(cv.CAP_PROP_FRAME_WIDTH), VID_CAP.get(cv.CAP_PROP_FRAME_HEIGHT)) # width by height
screen = pygame.display.set_mode(window_size)
bird_img = pygame.image.load("bird_sprite.png")
print(bird_img.get_height(), bird_img.get_width())
bird_img = pygame.transform.scale(bird_img, (bird_img.get_width() / 6, bird_img.get_height() / 6))
bird_frame = bird_img.get_rect()
bird_frame.x, bird_frame.y = window_size[0] // 6, window_size[1] // 2
pipe_frames = deque()
pipe_img = pygame.image.load("pipe_sprite_single.png")
starting_pipe_rect = pipe_img.get_rect()
space_between_pipes = 250

def addPipes():
    global pipe_frames, starting_pipe_rect
    top = starting_pipe_rect.copy()
    top.x, top.y = window_size[0], random.randint(120-1000, window_size[1]-120-space_between_pipes-1000)
    bottom = starting_pipe_rect.copy()
    bottom.x, bottom.y = window_size[0], top.y+1000+space_between_pipes
    pipe_frames.append([top, bottom])

def addMeshToFace(frame, results):
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_tesselation_style())
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_contours_style())
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_iris_connections_style())

# Game loop
prevChinY = None
gameClock = time.time()
stage = 1
pipeSpawnTimer = 0
T = 50
pipe_velocity = lambda: 700/T
level = 0
score = 0
didUpdateScore = False
game_is_running = True

with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:
    while True:
        if not game_is_running:
            text = pygame.font.SysFont("Helvetica Neue", 64).render('Game over!', True, (99, 245, 255))
            tr = text.get_rect()
            tr.center = (window_size[0]/2, window_size[1]/2)
            screen.blit(text, tr)
            pygame.display.update()
            pygame.time.wait(2000)
            VID_CAP.release()
            cv.destroyAllWindows()
            pygame.quit()
            sys.exit()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                VID_CAP.release()
                cv.destroyAllWindows()
                pygame.quit()
                sys.exit()

        ret, frame = VID_CAP.read()
        if not ret:
            print("Empty frame, continuing...")
            continue

        # Clear screen
        screen.fill((125, 220, 232))

        # Face mesh
        frame.flags.writeable = False
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = face_mesh.process(frame)
        frame.flags.writeable = True

        # Draw mesh
        # addMeshToFace(frame, results)
        if results.multi_face_landmarks and len(results.multi_face_landmarks) > 0:
            # 94 = Tip of nose
            # 152 = Lowest point on chin,
            chinY = results.multi_face_landmarks[0].landmark[94].y
            if prevChinY is None: prevChinY = chinY
            delta = (chinY - prevChinY) * 2500
            if abs(delta) < 5: delta = 0
            bird_frame.y += delta
            if bird_frame.top < 0: bird_frame.y = 0
            if bird_frame.bottom > window_size[1]: bird_frame.y = window_size[1] - bird_frame.height
            prevChinY = chinY

        # Mirror frame, swap axes because opencv != pygame
        frame = cv.flip(frame, 1).swapaxes(0, 1)

        # Update pipe positions
        for pf in pipe_frames:
            pf[0].x -= pipe_velocity()
            pf[1].x -= pipe_velocity()

        if len(pipe_frames) > 0 and pipe_frames[0][0].right < 0:
            del pipe_frames[0]

        # Update screen
        pygame.surfarray.blit_array(screen, frame)
        screen.blit(bird_img, bird_frame)
        checker = True
        for pf in pipe_frames:
            # Check if bird went through to update score
            if pf[0].left <= bird_frame.x <= pf[0].right:
                checker = False
                if not didUpdateScore:
                    score += 1
                    didUpdateScore = True
            # Update screen
            screen.blit(pipe_img, pf[1])
            screen.blit(pygame.transform.flip(pipe_img, 0, 1), pf[0])
        if checker: didUpdateScore = False

        text = pygame.font.SysFont("Helvetica Neue", 32).render(f'Stage {stage}', True, (99, 245, 255))
        tr = text.get_rect()
        tr.center = (100, 50)
        screen.blit(text, tr)
        text = pygame.font.SysFont("Helvetica Neue", 32).render(f'Score: {score}', True, (99, 245, 255))
        tr = text.get_rect()
        tr.center = (100, 100)
        screen.blit(text, tr)

        pygame.display.flip()

        # Check if bird is touching a pipe
        if any([bird_frame.colliderect(pf[0]) or bird_frame.colliderect(pf[1]) for pf in pipe_frames]):
            game_is_running = False

        # Adjust clock
        if pipeSpawnTimer == 0: addPipes()
        pipeSpawnTimer += 1
        if pipeSpawnTimer >= T: pipeSpawnTimer = 0
        if time.time() - gameClock >= 10:
            print(f"Changing time from t={T} to t={5/6*T}")
            T *= 5/6
            stage += 1
            gameClock = time.time()

