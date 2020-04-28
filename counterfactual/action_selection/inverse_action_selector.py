import torch
from counterfactual.action_selection.base_action_selector import BaseActionSelector


class InverseActionSelector(BaseActionSelector):
    def __init__(self, policy, invert_function=None, action_space=None):
        """
        :param policy: original policy to invert
        :param invert_function: function which takes an action and returns the inverted action
        """
        self.policy = policy
        # If we don't have a function to invert actions, negate the action.
        # Note: only works for continuous control action spaces and assumes
        if invert_function is None:
            if action_space is not None and hasattr(action_space, 'low'):
                invert_function = lambda x: torch.clamp(-x, min=action_space.low, max=action_space.high)
            else:
                invert_function = lambda x: -x
        self.invert_function = invert_function

    def forward(self, state):
        real_action = self.policy(state)
        inverted_action = self.invert_function(real_action)
        return inverted_action
