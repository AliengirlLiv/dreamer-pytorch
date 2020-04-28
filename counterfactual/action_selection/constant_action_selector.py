from counterfactual.action_selection.base_action_selector import BaseActionSelector


class ConstantActionSelector(BaseActionSelector):
    def __init__(self, action):
        """
        :param action: constant action to produce every timestep
        """
        self.action = action

    def forward(self, state):
        return self.action
