
# from homes.tp4618.Documents.bitbucket.SuperGlueThesis.external.glue-factory.gluefactory.models.matchers.loadedLightGlue import LoadedLightGlue

# /homes/tp4618/Documents/bitbucket/SuperGlueThesis/external/LightGlue/lightglue

# from /homes/tp4618/Documents/bitbucket/SuperGlueThesis/external/LightGlue/lightglue import SuperPoint, DISK, utils
# /homes/tp4618/Documents/bitbucket/SuperGlueThesis/external/glue-factory/gluefactory/models/matchers/loadedLightGlue.py
from .models.matchers.lightglue_pretrained import LoadedLightGlue
from .models.matchers.lightglue import LightGlue
from lightglue import SuperPoint, DISK
from lightglue.utils import load_image, rbd
from lightglue import viz2d
from pathlib import Path

import torch


"""
python -m gluefactory.visualiseTrainedModel
"""
torch.set_grad_enabled(False)
images = Path("/homes/tp4618/Documents/bitbucket/SuperGlueThesis/external/glue-factory/data/denseForest/ForestTrail/data/1_lc")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 'mps', 'cpu'

extractor = SuperPoint(max_num_keypoints=256).eval().to(device)  # load the extractor
matcher = LightGlue(features="superpoint").eval().to(device)

image0 = load_image(images / "lc_img_0.png")
image1 = load_image(images / "lc_img_1.png")

feats0 = extractor.extract(image0.to(device))
feats1 = extractor.extract(image1.to(device))
matches01 = matcher({"image0": feats0, "image1": feats1})
feats0, feats1, matches01 = [
    rbd(x) for x in [feats0, feats1, matches01]
]  # remove batch dimension

kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]

# print(m_kpts0, "\n\n", m_kpts1)

axes = viz2d.plot_images([image0, image1])
viz2d.plot_matches(m_kpts0, m_kpts1, color="lime", lw=0.2)
viz2d.add_text(0, f'Stop after {matches01["stop"]} layers', fs=20)

# Save the current figure
plot_path = "/homes/tp4618/Documents/bitbucket/SuperGlueThesis/external/glue-factory/plot1.png"
viz2d.save_plot(plot_path)
# plt.savefig(plot_path, bbox_inches='tight')
print(plot_path)

kpc0, kpc1 = viz2d.cm_prune(matches01["prune0"]), viz2d.cm_prune(matches01["prune1"])
viz2d.plot_images([image0, image1])
viz2d.plot_keypoints([kpts0, kpts1], colors=[kpc0, kpc1], ps=10)

# Save the second plot
plot_path2 = "/homes/tp4618/Documents/bitbucket/SuperGlueThesis/external/glue-factory/plot2.png"
# plt.savefig(plot_path2, bbox_inches='tight')
viz2d.save_plot(plot_path2)
print(plot_path2)