import gym
import pickle
import numpy as np
import cv2
env = gym.make("CartPole-v1")
observation = env.reset()

episodes = {"frames":[], "actions":[]}

# for 56 *56 episode num = 500
# for 28 * 28 episode num = 1000 
for _episode in range(1000):
    frames = []
    actions = []
    for _frame in range(30):
        action = env.action_space.sample() # your agent here (this takes random actions)
        frame = env.render(mode='rgb_array')
        observation, reward, done, info = env.step(action)
        
        img = frame
        img = img[150:350, 200:400]
        img = cv2.resize(img, (28,28))
        
        frames.append(img)
        actions.append(action)
    observation = env.reset()
    episodes["frames"].append(frames)
    episodes["actions"].append(actions)
    env.close()

data = [np.array(episodes["frames"]), np.array(episodes["actions"])]
print(data[0].shape, data[1].shape)
with open('cartpole_28.pickle', mode='wb') as f:
    pickle.dump(data, f)
