from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
import random
import pandas
import itertools


class LearningAgent(Agent):
    def __init__(self, env):
        super(LearningAgent, self).__init__(env)
        self.color = 'red'
        self.planner = RoutePlanner(self.env, self)
        valid_actions = Environment.valid_actions
        self.q_values = pandas.DataFrame(columns=valid_actions,
                                         index=itertools.product(['red', 'green'],
                                                                 valid_actions,  # 4 way intersection
                                                                 valid_actions,
                                                                 valid_actions,
                                                                 valid_actions)).fillna(3141.59265)
        self.deadline_tracker = []
        self.trip_reward = 0
        self.trip_number = 0
        self.rewards = []

    def get_current_state(self):
        inputs = self.env.sense(self)
        state = (inputs['light'], self.next_waypoint, inputs['oncoming'], inputs['left'], inputs['right'])
        return state

    def reset(self, destination=None):
        self.planner.route_to(destination)
        self.rewards.append(self.trip_reward)
        self.trip_reward = 0
        self.trip_number += 1

    def update(self, t):
        self.next_waypoint = self.planner.next_waypoint()
        deadline = self.env.get_deadline(self)
        self.deadline_tracker.append(self.trip_number) if deadline < 1 else 0
        self.state = self.get_current_state()
        action, q_value = self.explore_exploit(self.state)
        reward = self.env.act(self, action)
        self.trip_reward += reward
        self.update_q_value(reward, q_value, self.state, action)
        print "deadline = {}, action = {}, reward = {}, trip# = {}".format(deadline, action, reward, self.trip_number)

    def choose_action_and_q_value(self, state):
        row = self.q_values.loc[[state]]
        action = row.idxmax(axis=1)[0]
        q_value = row[action][0]
        return action, q_value

    def update_q_value(self, reward, q_value, state, action):
        discount_factor = 0.5
        learning_rate = 0.8
        new_state = self.get_current_state()
        new_q = self.choose_action_and_q_value(new_state)[1]
        new_q = (1 - learning_rate) * q_value + learning_rate * (reward + discount_factor * new_q)
        self.q_values.set_value(state, action, new_q)

    def explore_exploit(self, state):
        explore_chance = 1 / (self.trip_number + 0.9)
        if random.random() < explore_chance:
            action = random.choice(Environment.valid_actions)
            q_value = self.q_values.loc[[state]][action][0]
        else:
            action, q_value = self.choose_action_and_q_value(state)
        return action, q_value


def run():
    e = Environment()
    a = e.create_agent(LearningAgent)
    e.set_primary_agent(a, enforce_deadline=True)  # set agent to track
    sim = Simulator(e, update_delay=0.01)  # reduce update_delay to speed up simulation
    sim.run(n_trials=250)  # press Esc or close pygame window to quit
    print 'trip rewards: {}'.format(a.rewards)
    print 'Missed deadlines: {}'.format(a.deadline_tracker)


if __name__ == '__main__':
    run()
