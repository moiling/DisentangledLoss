import logging
import os

import torchvision


def save_images(out_dir, names, pred_trimaps_softmax, logger=logging.getLogger('utils')):
    """Save a batch of images."""
    trimap_path = os.path.join(out_dir, 'trimap')

    os.makedirs(trimap_path, exist_ok=True)

    # logger.debug(f'Saving {len(names)} images to {out_dir}')

    for idx, name in enumerate(names):
        if pred_trimaps_softmax is not None:
            trimap = pred_trimaps_softmax[idx]
            trimap = trimap.argmax(dim=0)
            trimap = trimap / 2.
            save_path = os.path.join(trimap_path, name)
            torchvision.utils.save_image(trimap, save_path)
