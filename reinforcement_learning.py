import numpy as np
import tensorflow as tf
import gym

# Parameters
EPISODES = 10000
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.99
EPSILON = 0.1

# Create the FrozenLake environment
env = gym.make('FrozenLake-v1')
num_actions = env.action_space.n
num_states = env.observation_space.n

# Deep Q-Network model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(num_states,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(num_actions, activation='linear')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
              loss='mean_squared_error')


# Function to select an action using epsilon-greedy strategy
def choose_action(state, epsilon):
    if np.random.uniform(0, 1) < epsilon:
        return np.random.randint(num_actions)
    else:
        Q_values = model.predict(np.array([state]))
        return np.argmax(Q_values)


# Function to perform the Q-learning update
def q_learning_update(state, action, reward, next_state, done):
    target = reward
    if not done:
        Q_next = model.predict(np.array([next_state]))
        target += DISCOUNT_FACTOR * np.max(Q_next)

    Q_values = model.predict(np.array([state]))
    Q_values[0][action] = target

    model.fit(np.array([state]), Q_values, epochs=1, verbose=0)


# Deep Q-Learning
@tf.function
def deep_q_learning():
    epsilon = EPSILON
    rewards = []

    for episode in range(EPISODES):
        state = env.reset()
        total_reward = 0

        while True:
            # Convert the integer state to one-hot encoded array and reshape it
            state_one_hot = np.zeros(num_states)
            state_one_hot[state] = 1
            state_one_hot = np.reshape(state_one_hot, (1, num_states))

            action = choose_action(state_one_hot, epsilon)
            next_state, reward, done, _ = env.step(action)
            total_reward += reward

            # Convert the integer next_state to one-hot encoded array and reshape it
            next_state_one_hot = np.zeros(num_states)
            next_state_one_hot[next_state] = 1
            next_state_one_hot = np.reshape(next_state_one_hot, (1, num_states))

            q_learning_update(state_one_hot, action, reward, next_state_one_hot, done)

            state = next_state

            if done:
                rewards.append(total_reward)
                break

        # Decay epsilon over episodes
        epsilon = max(0.1, epsilon * 0.999)

        # Print progress
        if episode % 100 == 0:
            print(f"Episode {episode}, Average Reward: {np.mean(rewards[-100:])}")


if __name__ == "__main__":
    deep_q_learning()
