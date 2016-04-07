import unittest
import agent
from environment import Environment


class TestAgent(unittest.TestCase):
    green_forward_state = ('green', 'forward', None, None, None)
    red_right_state = ('red', 'right', None, None, None)
    red_forward_state = ('red', 'forward', None, None, None)

    def setUp(self):
        env = Environment()
        self.agent = env.create_agent(agent.LearningAgent)
        self.agent.q_values[self.agent.q_values > 0] = 0.1

    def testChooseActionShouldReturnActionWithHighestQValue(self):
        self.agent.q_values.set_value(self.green_forward_state, 'left', 10)
        self.assertEqual(self.agent.choose_action_and_q_value(self.green_forward_state)[0], 'left')

    def testChooseActionShouldWorkWithMultipleSameValues(self):
        self.agent.q_values.set_value(self.green_forward_state, 'left', 10)
        self.agent.q_values.set_value(self.green_forward_state, 'right', 10)
        self.assertEqual(self.agent.choose_action_and_q_value(self.green_forward_state)[0], 'left')

    def testChooseActionShouldNotCareAboutOtherStates(self):
        self.agent.q_values.set_value(self.green_forward_state, 'left', 10)
        self.agent.q_values.set_value(self.red_forward_state, 'right', 100)
        self.assertEqual(self.agent.choose_action_and_q_value(self.green_forward_state)[0], 'left')
        self.assertNotEqual(self.agent.choose_action_and_q_value(self.green_forward_state)[0], 'right')

    def testChooseActionShouldWorkWithNegativeValues(self):
        self.agent.q_values.set_value(self.green_forward_state, 'left', 10.01)
        self.agent.q_values.set_value(self.green_forward_state, 'right', -10)
        self.assertEqual(self.agent.choose_action_and_q_value(self.green_forward_state)[0], 'left')

    def testUpdateShouldUpdateQValues(self):
        old_q_value = 5.5
        action = 'left'
        self.agent.q_values.set_value(self.green_forward_state, action, old_q_value)
        self.agent.update_q_value(1, old_q_value, self.green_forward_state, action)
        new_q_value = self.agent.q_values.loc[[self.red_right_state]][action][0]
        self.assertNotEqual(new_q_value, old_q_value)

    def testExploreExploitShouldWorkOnFirstTrip(self):
        self.agent.trip_number = 0
        self.assertTrue(self.agent.explore_exploit(self.green_forward_state)[0] in Environment.valid_actions)
        self.assertTrue(self.agent.explore_exploit(self.green_forward_state)[1] > 0)

    def testChooseActionShouldReturnCorrectQValue(self):
        self.agent.q_values.set_value(self.green_forward_state, 'left', 10.123)
        self.assertEqual(self.agent.choose_action_and_q_value(self.green_forward_state)[0], 'left')
        self.assertEqual(self.agent.choose_action_and_q_value(self.green_forward_state)[1], 10.123)

    def testExploreExploitShouldWorkOnBillionthTrip(self):
        self.agent.trip_number = 1000000000
        self.agent.q_values.set_value(self.green_forward_state, 'left', 10.123)
        self.assertEqual(self.agent.choose_action_and_q_value(self.green_forward_state)[0], 'left')
        self.assertEqual(self.agent.choose_action_and_q_value(self.green_forward_state)[1], 10.123)


if __name__ == '__main__':
    unittest.main()
