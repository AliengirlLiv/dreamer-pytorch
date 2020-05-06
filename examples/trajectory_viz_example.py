import imageio

from counterfactuals.trajectory_visualization import side_by_side_visualization
from counterfactuals.dataset import TrajectoryDataset


# Example of how to generate video from trajectory datasets
dataset_pretrained = TrajectoryDataset()
dataset_pretrained.load("/Users/ericweiner/Downloads/data/pretrained")
dataset_counterfactual = TrajectoryDataset()
dataset_counterfactual.load("/Users/ericweiner/Downloads/data/counterfactual")
FPS = 3
for i, (pt, ct) in enumerate(
    zip(dataset_pretrained.trajectory_generator(), dataset_counterfactual.trajectory_generator())
):
    pre_video, cf_video = side_by_side_visualization(pt, ct)
    imageio.mimsave(f'cf_traj{i}.gif', cf_video, fps=FPS)
    imageio.mimsave(f'pre_traj{i}.gif', pre_video, fps=FPS)
    
    # To write to an mp4, but we are using gifs right now!
    # out = cv2.VideoWriter(
    #     f"tmp{i}.mp4",
    #     cv2.VideoWriter_fourcc(*"mp4v"),
    #     FPS,
    #     (cf_video[0].shape[1], cf_video[0].shape[0]),
    # )
    
    
    # for frame in cf_video:
    #     out.write(frame)
    # out.release()