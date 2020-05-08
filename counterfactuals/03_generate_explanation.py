import argparse
import os
import imageio

from counterfactuals.dataset import TrajectoryDataset, leading_zeros
from counterfactuals.trajectory_visualization import side_by_side_visualization, overlay_visualization


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default=os.path.join('..', 'data'))
    parser.add_argument('--save-dir', default=os.path.join('..', 'videos'))
    parser.add_argument('--fps', default=1.5)
    args = parser.parse_args()

    for bot_dir in os.listdir(args.data_dir):
        original_path = os.path.join(args.data_dir, bot_dir, 'original')
        original_dataset = TrajectoryDataset(original_path)
        original_dataset.load()
        cf_path = os.path.join(args.data_dir, bot_dir, 'counterfactual')
        user_dataset = TrajectoryDataset(os.path.join(cf_path, 'user'))
        user_dataset.load()
        for cf_data_dir in os.listdir(cf_path):
            if 'user' in cf_data_dir:
                continue
            counterfactual_dataset = TrajectoryDataset(os.path.join(cf_path, cf_data_dir))
            counterfactual_dataset.load()
            for traj_orig, traj_cf, user_cf in zip(original_dataset.trajectory_generator(),
                                          counterfactual_dataset.trajectory_generator(),
                                          user_dataset.trajectory_generator()):
                episode = traj_orig.episode
                original_video, cf_video, original_cont, cf_cont = overlay_visualization(traj_orig, traj_cf, user_cf)
                name = f'{leading_zeros(episode)}.gif'
                combination_path = os.path.join(args.save_dir, 'original-' + bot_dir, 'explanation-' + cf_data_dir)
                original_path = os.path.join(combination_path, "original")
                explanation_path = os.path.join(combination_path, "explanation")
                original_cont_path = os.path.join(combination_path, "original-cont")
                explanation_cont_path = os.path.join(combination_path, "explanation-cont")

                for path in [combination_path, original_path, explanation_path, original_cont_path, explanation_cont_path]:
                    if not os.path.exists(path):
                        os.makedirs(path)
                imageio.mimsave(os.path.join(original_path, name), original_video, fps=args.fps)
                imageio.mimsave(os.path.join(explanation_path, name), cf_video, fps=args.fps)
                imageio.mimsave(os.path.join(original_cont_path, name), original_cont, fps=args.fps)
                imageio.mimsave(os.path.join(explanation_cont_path, name), cf_cont, fps=args.fps)


if __name__ == '__main__':
    main()
