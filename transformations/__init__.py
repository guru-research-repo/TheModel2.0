from transformations.salience_crop import SalienceCrop
from transformations.salience import SalienceSampling
# from transformations.salience_old import SalienceSampling as SalienceSamplingOld
from transformations.log_polar import LogPolar
# from transformations.log_polar_old import LogPolar as LogPolarOld
from transformations.n_random_crop import NRandomCrop
from transformations.compose import Compose
from transformations.resize import Resize
# from transformations.foveation_old import Foveation as FoveationOld
from transformations.foveation import Foveation
from transformations.identity import Identity
from transformations.replicate import Replicate
from transformations.random_rotate import RandomRotate
from transformations.hemisphere_crop import HemisphereCrop

from transformations.pipeline import Pipeline

from transformations.simclr_pipeline import SimCLREvalDataTransform, SimCLRTrainDataTransform
from transformations.hemisphere_pipeline import HemispherePipeline
from transformations.get_pipeline import get_pipeline
from transformations.get_eval_pipeline import get_eval_pipeline
