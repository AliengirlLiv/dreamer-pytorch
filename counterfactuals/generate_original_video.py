import argparse
import os
import imageio

from counterfactuals.dataset import TrajectoryDataset, leading_zeros
from counterfactuals.trajectory_visualization import side_by_side_visualization, overlay_visualization


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load-dir', required=True)
    parser.add_argument('--save-dir', required=True)
    parser.add_argument('--fps', default=1.5)
    args = parser.parse_args()
    dataset = TrajectoryDataset(args.load_dir)
    dataset.load()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    for traj_orig in dataset.trajectory_generator():
        episode = traj_orig.episode
        original_video, _ = side_by_side_visualization(traj_orig, traj_orig)
        name = f'{leading_zeros(episode)}.gif'
        path = os.path.join(args.save_dir, name)
        imageio.mimsave(path, original_video, fps=args.fps)
    dataset.load()


if __name__ == '__main__':
    main()
