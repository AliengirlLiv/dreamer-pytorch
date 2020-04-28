class BaseActionSelector:

    def forward(self, state):
        """
        Take in a state, return an action.  Override this in all child classes.
        """
        raise NotImplementedError