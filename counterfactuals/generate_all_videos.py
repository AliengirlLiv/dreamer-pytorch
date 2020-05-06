import imageio
import pathlib

from counterfactuals.trajectory_visualization import side_by_side_visualization
from counterfactuals.dataset import TrajectoryDataset

FPS = 3
file = pathlib.Path("/home/olivia/Downloads/data")
for bot_files in file.iterdir(): # Each subfolder contains all data with one bot as the original policy
    original_dataset_path = bot_files.joinpath("original")
    cf_trajs = bot_files.joinpath("counterfactual/user")
    for cf_dataset_path in cf_trajs.iterdir(): # Each subfolder contains all data with a particular other policy as the continuation
        dataset_pretrained = TrajectoryDataset()
        dataset_pretrained.load(original_dataset_path)
        dataset_counterfactual = TrajectoryDataset()
        dataset_counterfactual.load(cf_dataset_path)

        for i, (pt, ct) in enumerate(
                zip(dataset_pretrained.trajectory_generator(), dataset_counterfactual.trajectory_generator())
        ):
            pre_video, cf_video = side_by_side_visualization(pt, ct)
            imageio.mimsave(f'cf_traj{i}.gif', cf_video, fps=FPS)
            imageio.mimsave(f'pre_traj{i}.gif', pre_video, fps=FPS)  # TODO: Save these in a file structure similar to the initial one.

