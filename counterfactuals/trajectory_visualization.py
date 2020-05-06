import cv2
import numpy as np
import copy
from gym_minigrid.minigrid import TILE_PIXELS

from counterfactuals.dataset import Trajectory
from counterfactuals.color_modified_gridworld import render_agent_tile

YELLOW = (255, 255, 0)
PURPLE = (255, 0, 255)

def add_centered_text(
    img: np.ndarray,
    text: str,
    color: tuple = (255, 255, 255),
    scale: int = 1,
    top_padding: int = 10,
):
    thickness = 1
    font = cv2.FONT_HERSHEY_SIMPLEX
    textsize = cv2.getTextSize(text, font, scale, thickness)[0]
    textcoords = ((img.shape[1] - textsize[0]) // 2, textsize[1] + top_padding)
    cv2.putText(img, text, textcoords, font, scale, color, thickness, cv2.LINE_AA)

def add_mission_text(img: np.ndarray, text:str, scale=0.7):
    """
    Add boarder at the bottom with mission text
    """
    thickness = 1
    color = (255, 255, 255)
    font = cv2.FONT_HERSHEY_SIMPLEX
    textsize = cv2.getTextSize(text, font, scale, thickness)[0]
    img = cv2.copyMakeBorder(
            img,
            top=0,
            bottom=textsize[1] + 20,
            left=0,
            right=0,
            borderType=cv2.BORDER_CONSTANT,
            value=[0, 0, 0]
    )
    textcoords = ((img.shape[1] // 2 - textsize[0] // 2, img.shape[0] - textsize[1]))
    cv2.putText(img, text, textcoords, font, scale, color, thickness, cv2.LINE_AA)
    return img


def gray_and_blur(img: np.ndarray):
    """
    Filter applied to an image that is not active
    """
    g_img = np.repeat(np.expand_dims(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), 2), 3, 2)
    kernel = np.ones((15, 15), np.float32) / 225
    dst = cv2.filter2D(g_img, -1, kernel)
    return dst


def make_img(
    state,
    text: str,
    text_scale: float = 0.6,
    downscale_factor: int = 1,
    active: bool = True,
):
    """
    :param state: State to render
    :param text: Text to write in center
    :param text_scale: Size of the text, usually 0.6 or 1.2
    :param downscale_factor: Most env renders are too big, so downsize it
    :param active: Should this be grayed and blurred?
    """
    frame = state.render(mode="rgb_array")
    new_size = (frame.shape[1] // downscale_factor, frame.shape[0] // downscale_factor)
    frame = cv2.resize(frame, new_size)
    add_centered_text(frame, text, scale=text_scale)
    if not active:
        return gray_and_blur(frame)
    return frame


def compose_frames(
    big_frame: np.ndarray, top_right: np.ndarray, bottom_right: np.ndarray
):
    """
                            |--------------|--------|
                            |              |        |
    Makes frames like this: |              |--------|
                            |              |        |
                            |______________|________|
    """
    stacked_frames = np.append(top_right, bottom_right, 0)
    return np.append(big_frame, stacked_frames, 1)


def update_render(img, agent_pos, agent_dir, color, tile_size=TILE_PIXELS):
    """
    """
    tile_img = render_agent_tile(agent_dir, color)

    i, j = agent_pos
    ymin = j * tile_size
    ymax = (j + 1) * tile_size
    xmin = i * tile_size
    xmax = (i + 1) * tile_size
    img[ymin:ymax, xmin:xmax, :] = tile_img

def overlay_cf(img1, img2):
    return np.max(np.stack([img1, img2], axis=0), axis=0)

def overlay_visualization(original: Trajectory, counterfactual: Trajectory, user_states: Trajectory):
    """
    Creates a visualization to view how the behavior of the counterfactual
    trajectory differs from the original, or pretrained trajectory.
    """
    divergent_step = counterfactual.step[0]
    o_divergent_state = copy.deepcopy(original.state[divergent_step])
    counterfactual_step = 0
    combined_video = []
    pretrained_video = []
    pretrained_continuation = []
    cf_continuation = []
    for state, step in zip(original.state, original.step):

        diverged = step >= divergent_step

        # Construct Pretrained Frame
        pretrained_img = make_img(state, "Original Trajectory", text_scale=1.2)
        if hasattr(state, 'mission'):
            pretrained_img = add_mission_text(pretrained_img, state.mission)
        pretrained_video.append(pretrained_img)
        if diverged:
            pretrained_continuation.append(pretrained_img)
        if step == original.step[-1]:
            add_centered_text(pretrained_img, "DONE", top_padding=200, scale=2.4)
            pretrained_video.extend([pretrained_img] * 3)

        # Construct Pretrained + Counterfactual Frame
        original_img = make_img(state, "Original Trajectory", text_scale=1.2, active=True)
        c_state = counterfactual.state[counterfactual_step]
        if diverged:
            color = YELLOW if step in user_states.step else PURPLE
            cf_image = make_img(c_state, "", text_scale=1.2)
            # Color the counterfactual agent differently
            update_render(cf_image, c_state.agent_pos, c_state.agent_dir, color)
            combined_image = overlay_cf(original_img, cf_image)

        else:
            combined_image = original_img
        if hasattr(state, 'mission'):
            combined_image = add_mission_text(combined_image, state.mission)

        # Some trajectories had different lengths, so this preserves
        # the counterfactual state if this trajectory ends first
        if counterfactual_step < len(counterfactual.state) - 1:
            if step >= divergent_step:
                counterfactual_step += 1
            c_state.close()
        elif counterfactual_step == len(counterfactual.state) - 1:
            add_centered_text(combined_image, "Done with counterfactual", top_padding=100)

        # And this checks if the counterfactual trajectory is longer
        if step == original.step[-1]:
            while counterfactual_step < len(counterfactual.step) - 1:
                c_state = counterfactual.state[counterfactual_step]
                cf_image = make_img(c_state, "", text_scale=1.2)
                update_render(cf_image, c_state.agent_pos, c_state.agent_dir, PURPLE)
                combined_image = overlay_cf(original_img, cf_image)
                add_centered_text(combined_image, "Done with original", top_padding=200)
                if hasattr(state, 'mission'):
                    combined_image = add_mission_text(combined_image, state.mission)
                combined_video.append(combined_image)
                c_state.close()
                counterfactual_step += 1
            c_state = counterfactual.state[counterfactual_step]
            cf_image = make_img(c_state, "", text_scale=1.2)
            update_render(cf_image, c_state.agent_pos, c_state.agent_dir, PURPLE)
            combined_image = overlay_cf(original_img, cf_image)
            if hasattr(state, 'mission'):
                combined_image = add_mission_text(combined_image, state.mission)
            add_centered_text(combined_image, "Done with original", top_padding=200)
            add_centered_text(combined_image, "Done with counterfactual", top_padding=100)
        combined_video.append(combined_image)

        c_state.close()
        state.close()

        # Save completion policy individually too
        cf_image = make_img(c_state, "", text_scale=1.2)
        if hasattr(state, 'mission'):
            cf_image = add_mission_text(cf_image, state.mission)
        cf_continuation.append(cf_image)
        if counterfactual_step == counterfactual.step[-1]:
            add_centered_text(cf_image, "DONE", top_padding=200, scale=2.4)
            cf_continuation.extend([cf_image] * 3)

    combined_video.extend([combined_image] * 5)



    o_divergent_state.close()
    return pretrained_video, combined_video, pretrained_continuation, cf_continuation



def side_by_side_visualization(original: Trajectory, counterfactual: Trajectory):
    """
    Creates a visualization to view how the behavior of the counterfactual
    trajectory differs from the original, or pretrained trajectory.
    """
    divergent_step = counterfactual.step[0]
    o_divergent_state = copy.deepcopy(original.state[divergent_step])
    counterfactual_step = 0
    cf_video = []
    pretrained_video = []
    for state, step in zip(original.state, original.step):
        
        # Construct Pretrained Frame
        pretrained_img = make_img(state, "Original Trajectory", text_scale=1.2)
        if hasattr(state, 'mission'):
            pretrained_img = add_mission_text(pretrained_img, state.mission)
        pretrained_video.append(pretrained_img)
        if step == original.step[-1]:
            add_centered_text(pretrained_img, "DONE", top_padding=200, scale=2.4)
            pretrained_video.extend([pretrained_img] * 3)

        # Construct Pretrained + Counterfactual Frame
        # We are never blurring the frame on the left
        not_diverged = step <= divergent_step
        s = state if not_diverged else o_divergent_state
        big_frame = make_img(s, "Original Trajectory", text_scale=1.2, active=True)
        if not not_diverged:
            big_frame = big_frame // 2
        

        # We do this to keep the big frame unblurred for one more step
        diverged = step >= divergent_step
        s = state if diverged else o_divergent_state
        o_divergent_img = make_img(
                s, "Original Trajectory", downscale_factor=2, active=diverged,
            )

        c_state = counterfactual.state[counterfactual_step]
        c_divergent_img = make_img(
            c_state, "Counterfactual Trajectory", downscale_factor=2, active=diverged
        )
        # Some trajectories had different lengths, so this preserves
        # the counterfactual state if this trajectory ends first
        if counterfactual_step < len(counterfactual.state) - 1:
            if step >= divergent_step:
                counterfactual_step += 1
            c_state.close()
        elif counterfactual_step == len(counterfactual.state) - 1:
            add_centered_text(c_divergent_img, "DONE", top_padding=100)

        # And this checks if the counterfactual trajectory is longer
        if step == original.step[-1]:
            add_centered_text(o_divergent_img, "DONE", top_padding=100)
            while counterfactual_step < len(counterfactual.step) - 1:
                c_state = counterfactual.state[counterfactual_step]
                c_divergent_img = make_img(
                    c_state,
                    "Counterfactual Trajectory",
                    downscale_factor=2,
                    active=diverged,
                )

                f = compose_frames(big_frame, o_divergent_img, c_divergent_img)
                if hasattr(state, 'mission'):
                    f = add_mission_text(f, state.mission)
                cf_video.append(f)
                c_state.close()
                counterfactual_step += 1
            c_state = counterfactual.state[counterfactual_step]
            c_divergent_img = make_img(
                c_state,
                "Counterfactual Trajectory",
                downscale_factor=2,
                active=diverged,
            )
            add_centered_text(c_divergent_img, "DONE", top_padding=100)
        f = compose_frames(big_frame, o_divergent_img, c_divergent_img)
        if hasattr(state, 'mission'):
            f = add_mission_text(f, state.mission)
        cf_video.append(f)  

        c_state.close()
        state.close()
    cf_video.extend([f] * 5)
    o_divergent_state.close()
    return pretrained_video, cf_video
