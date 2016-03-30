from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
import random
import pandas
import itertools


class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        # sets self.env = env, state = None, next_waypoint = None, and a default color
        super(LearningAgent, self).__init__(env)
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        valid_actions = Environment.valid_actions
        self.q_values = pandas.DataFrame(columns=valid_actions,
                                         index=itertools.product(['red', 'green'], valid_actions)).fillna(3.14159265)
        self.deadline_tracker = 0
        self.trip_reward = 0
        self.trip_number = 0

    def reset(self, destination=None):
        self.planner.route_to(destination)
        self.trip_reward = 0
        self.trip_number += 1

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)
        self.deadline_tracker += deadline if deadline <= 1 else 0
        self.state = (inputs['light'], self.next_waypoint)

        action, q_value = self.explore_exploit(self.state)

        # Execute action and get reward
        reward = self.env.act(self, action)
        self.trip_reward += reward
        self.update_q_value(reward, q_value, self.state, action)
        print "deadline = {}, inputs = {}, action = {}, reward = {}, fails = {}, total reward = {}, trip# = {}" \
            .format(deadline, inputs, action, reward, self.deadline_tracker, self.trip_reward, self.trip_number)

    def choose_action_and_q_value(self, state):
        row = self.q_values.loc[[state]]
        action = row.idxmax(axis=1)[0]
        q_value = row[action][0]
        return action, q_value

    def update_q_value(self, reward, q_value, state, action):
        discount_factor = 0.55
        learning_rate = 0.98

        new_state = (self.env.sense(self)['light'], self.next_waypoint)
        new_q = self.choose_action_and_q_value(new_state)[1]

        new_q = q_value + learning_rate * (reward + discount_factor * new_q - q_value)
        self.q_values.set_value(new_state, action, new_q)

        q_value += learning_rate * (reward + discount_factor * new_q - q_value)
        self.q_values.set_value(state, action, q_value)

    def explore_exploit(self, state):
        explore_chance = 1 / (self.trip_number + 0.9)
        if random.random() < explore_chance:
            action = random.choice(Environment.valid_actions)
            q_value = self.q_values.loc[[state]][action][0]
        else:
            action, q_value = self.choose_action_and_q_value(state)
        return action, q_value


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # set agent to track

    # Now simulate it
    sim = Simulator(e, update_delay=0.01)  # reduce update_delay to speed up simulation
    sim.run(n_trials=100)  # press Esc or close pygame window to quit


if __name__ == '__main__':
    run()
