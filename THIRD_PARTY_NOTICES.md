# Third-party source notes

The EGNN modules under `src/modules/egnn_pytorch*.py` and the molecular-surface
utilities under `scripts/docker_surface/{geometry_processing,helper}.py` are
copied from the existing SurfPDI-v2 project, with their existing comments.

Their source lineage includes the EGNN PyTorch implementation family and
dMaSIF-derived surface processing. The EGNN file also contains a link to a
lucidrains/se3-transformer-pytorch helper. The copied project did not contain
dedicated upstream license files beside these modules.

Before publication, identify the exact upstream versions used and preserve the
applicable attribution and license texts. This file does not assign a license
to third-party code or claim a verified upstream commit.

Reference locations for the source-lineage review:

- https://github.com/lucidrains/egnn-pytorch
- https://github.com/lucidrains/se3-transformer-pytorch
- https://github.com/FreyrS/dMaSIF
