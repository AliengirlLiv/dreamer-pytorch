import argparse

def build_and_run(n_videos, n_frames, n_samples, sample_from_env, state_selection, action_selection,
                  real_policy_path, adv_policy_path, dataset_path, save_path):
    asdf = 3
    # create agent
    # create env, if applicable
    # Create counterfactual policy (based on action_selection argument value)
    # load dataset from path, if available.  Otherwise, create new dataset class
    # Fill dataset with n_samples samples.  If sample_from_env is true, sample from a real environment.  Otherwise use the learned model.
    # save dataset to dataset_path, if provided
    # get state indices
    # TODO: this part feels like it should be it's own file
    # for each index:
    #   obtain the episode data for that index
    #   Extract previous k timesteps (or until beginning or trajectory) of video frames leading up to each index
    #   for t in n_frames steps:
    #      apply real and counterfactual policy starting from the start state
    #      # Store data points in a new Dataset
    #      # call save_videos on a list of both datasets.




if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Add in flags for all the args we need to pass to build_and_run
    args = parser.parse_args()
    build_and_run(
        n_videos=args.n_videos,
        n_frames=args.n_frames,
        n_samples=args.n_samples,
        sample_from_env=args.sample_from_env,
        state_selection = args.state_selection,
        action_selection=args.action_selection,
        real_policy_path=args.real_policy_path,
        adv_policy_path=args.adv_policy_path,
        dataset_path=args.dataset_path,
        save_path=args.save_path
    )



# TODO: TONIGHT Have
#       - main function which runs with a single set of arguments
#       - each function prints what it's doing at the moment
#       - create project board with tasks
#       - push and commit