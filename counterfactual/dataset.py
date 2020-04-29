
class Episode:
    def __init__(self, keys):
        pass

    def add_timestep(self, timestep_dict):
        pass

    def get_key(self): # Get all the elements from one of the dictionary keys (e.g. just images)
        pass

    def get_timestep(self, index):
        pass


class Dataset:  # TBH we could probably just use the dreamer buffer class here!
    def __init__(self):
        self.episodes = [] # List which we will populate with episodes

    def save_dataset(self, path):
        pass

    def add_episode(self, episode):
        pass

    def get_episode(self, index):
        pass

    def get_frame(self, episode_index, frame_index):
        pass