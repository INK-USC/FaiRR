import numpy as np, sys, os, shutil, struct, argparse, csv, math, uuid, jsonlines, types, pathlib, getpass, re, random, nltk, itertools, traceback
import time, socket, logging, itertools, json
import pickle5 as pickle

from argparse import ArgumentParser
from datetime import datetime, timedelta
from tqdm import tqdm
from typing import Optional
from functools import lru_cache
from collections import defaultdict as ddict
from collections import OrderedDict, Counter
from typing import Dict, List, NamedTuple, Optional
from configparser import ConfigParser
from copy import deepcopy
from pprint import pprint
from pprint import pformat

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.utils.data import DataLoader, Dataset
from torch.nn.init import xavier_normal_, kaiming_uniform_, xavier_uniform_
from torch.optim import Adam

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from sklearn.metrics import f1_score

from nltk.tokenize import word_tokenize, sent_tokenize

import datasets
from transformers import (
	AdamW,
	Adafactor,
	AutoModelForSequenceClassification,
	AutoModelForMultipleChoice,
	T5ForConditionalGeneration,
	AutoModel,
	AutoConfig,
	AutoTokenizer,
	T5Tokenizer,
	get_scheduler,
	get_linear_schedule_with_warmup,
	glue_compute_metrics
)

import hydra
from omegaconf import DictConfig, OmegaConf
from hydra import compose, initialize
