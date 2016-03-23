import unittest
import agent
from environment import Environment


class TestAgent(unittest.TestCase):
    def setUp(self):
        env = Environment()
        self.agent = env.create_agent(agent.LearningAgent)

    def testChooseActionShouldReturnActionWithHighestQValue(self):
        state = ('green', 'forward')
        self.agent.q_values.set_value(state, 'left', 10)
        self.assertEqual(self.agent.choose_action_and_q_value(state)[0], 'left')

    def testChooseActionShouldWorkWithMultipleSameValues(self):
        state = ('green', 'forward')
        self.agent.q_values.set_value(state, 'left', 10)
        self.agent.q_values.set_value(state, 'right', 10)
        self.assertEqual(self.agent.choose_action_and_q_value(state)[0], 'left')

    def testChooseActionShouldNotCareAboutOtherStates(self):
        state = ('green', 'forward')
        state1 = ('red', 'forward')
        self.agent.q_values.set_value(state, 'left', 10)
        self.agent.q_values.set_value(state1, 'right', 100)
        self.assertEqual(self.agent.choose_action_and_q_value(state)[0], 'left')
        self.assertNotEqual(self.agent.choose_action_and_q_value(state)[0], 'right')

    def testChooseActionShouldWorkWithNegativeValues(self):
        state = ('green', 'forward')
        self.agent.q_values.set_value(state, 'left', 10.01)
        self.agent.q_values.set_value(state, 'right', -10)
        self.assertEqual(self.agent.choose_action_and_q_value(state)[0], 'left')

    def testUpdateShouldUpdateQValues(self):
        state = ('green', 'forward')
        old_q_value = 5.5
        action = 'left'
        self.agent.q_values.set_value(state, action, old_q_value)
        self.agent.update_q_value(1, old_q_value, state, action)
        new_q_value = self.agent.q_values.loc[[state]][action][0]
        self.assertNotEqual(new_q_value, old_q_value)


if __name__ == '__main__':
    unittest.main()
