import argparse
import os
import imageio

from counterfactuals.dataset import TrajectoryDataset, leading_zeros
from counterfactuals.trajectory_visualization import side_by_side_visualization


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default=os.path.join('..', 'data'))
    parser.add_argument('--save-dir', default=os.path.join('..', 'videos'))
    parser.add_argument('--fps', default=1)
    args = parser.parse_args()

    for bot_dir in os.listdir(args.data_dir):
        original_path = os.path.join(args.data_dir, bot_dir, 'original')
        original_dataset = TrajectoryDataset(original_path)
        original_dataset.load()
        cf_path = os.path.join(args.data_dir, bot_dir, 'counterfactual')
        for cf_data_dir in os.listdir(cf_path):
            if 'user' in cf_data_dir:
                continue
            counterfactual_dataset = TrajectoryDataset(os.path.join(cf_path, cf_data_dir))
            counterfactual_dataset.load()
            for traj_orig, traj_cf in zip(original_dataset.trajectory_generator(),
                                          counterfactual_dataset.trajectory_generator()):
                episode = traj_orig.episode
                original_video, cf_video = side_by_side_visualization(traj_orig, traj_cf)
                original_name = f'{leading_zeros(episode)}-original.gif'
                explanation_name = f'{leading_zeros(episode)}-explanation.gif'
                combination_path = os.path.join(args.save_dir, 'original-' + bot_dir, 'explanation-' + cf_data_dir)
                if not os.path.exists(combination_path):
                    os.makedirs(combination_path)
                imageio.mimsave(os.path.join(combination_path, original_name), original_video, fps=args.fps)
                imageio.mimsave(os.path.join(combination_path, explanation_name), cf_video, fps=args.fps)


if __name__ == '__main__':
    main()
