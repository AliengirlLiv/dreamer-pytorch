from gym_minigrid.rendering import fill_coords, rotate_fn, point_in_triangle, point_in_rect, downsample, highlight_img
from gym_minigrid.minigrid import TILE_PIXELS
import math
import numpy as np


tile_cache = {}

def render_agent_tile(
    agent_dir,
    color,
    tile_size=TILE_PIXELS,
    subdivs=3,
):
    """
    Render a tile and cache the result
    """

    # Hash map lookup key for the cache
    key = (agent_dir, color, tile_size)

    if key in tile_cache:
        return tile_cache[key]

    img = np.zeros(shape=(tile_size * subdivs, tile_size * subdivs, 3), dtype=np.uint8)

    # Draw the grid lines (top and left edges)
    fill_coords(img, point_in_rect(0, 0.031, 0, 1), (100, 100, 100))
    fill_coords(img, point_in_rect(0, 1, 0, 0.031), (100, 100, 100))

    # Overlay the agent on top
    tri_fn = point_in_triangle(
        (0.12, 0.19),
        (0.87, 0.50),
        (0.12, 0.81),
    )

    # Rotate the agent based on its direction
    tri_fn = rotate_fn(tri_fn, cx=0.5, cy=0.5, theta=0.5*math.pi*agent_dir)
    fill_coords(img, tri_fn, color)

    highlight_img(img)

    # Downsample the image to perform supersampling/anti-aliasing
    img = downsample(img, subdivs)

    # Cache the rendered tile
    tile_cache[key] = img

    return img