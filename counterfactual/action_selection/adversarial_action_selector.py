from counterfactual.action_selection.base_action_selector import BaseActionSelector

class AdversarialAction(BaseActionSelector):
    def __init__(self, policy):
        """
        Select actions by calling an adversarial/alternative.
        :param policy: adversarial policy to run
        """
        self.policy = policy

    def forward(self, state):
        return self.policy(state)