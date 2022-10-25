import sys
root_dir = r'C:\Users\intern\Downloads\til-final-2022.1.1-best\til-final-2022.1.1-lower'
root_dir_nlp = r'C:\Users\intern\Downloads\til-final-2022.1.1-best\til-final-2022.1.1-lower\installs-new-nlp'
sys.path.insert(1, root_dir_nlp)
from typing import Iterable, List
from tilsdk.localization.types import *
# import onnxruntime as ort
# import numpy as np
# from pathlib import Path
# from tqdm import tqdm
# import torchaudio
# from sklearn.model_selection import train_test_split
# import os
# import sys
# from transformers import AutoConfig, Wav2Vec2Processor, Wav2Vec2FeatureExtractor
# import torch.nn.functional as F
# import soundfile as sf
# from io import BytesIO
# import librosa
# import pandas as pd
# from dataclasses import dataclass
# from typing import Optional, Tuple
# import torch
# from transformers.file_utils import ModelOutput
# import torch.nn as nn
# from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
# from transformers.models.hubert.modeling_hubert import (
#     HubertPreTrainedModel,
#     HubertModel
# )
# from dataclasses import dataclass
# from typing import Dict, List, Optional, Union
# import transformers
# from transformers import Wav2Vec2Processor

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Device: {device}")
# label_list = ['angry', 'fear', 'happy', 'neutral', 'sad']
# num_labels = len(label_list)
# load_pretrainedmodel_path= r'C:\Users\intern\til-final-2022.1.1-lower\stubs\model\model_checkpoints_13epochs'
# load_pretrainedprocessor_path = r'C:\Users\intern\til-final-2022.1.1-lower\stubs\model\processor_checkpoints_13epochs'
# pooling_mode = "mean"
# is_regression = False

# @dataclass
# class SpeechClassifierOutput(ModelOutput):
#     loss: Optional[torch.FloatTensor] = None
#     logits: torch.FloatTensor = None
#     hidden_states: Optional[Tuple[torch.FloatTensor]] = None
#     attentions: Optional[Tuple[torch.FloatTensor]] = None


# class HubertClassificationHead(nn.Module):
#     """Head for hubert classification task."""

#     def __init__(self, config):
#         super().__init__()
#         self.dense = nn.Linear(config.hidden_size, config.hidden_size)
#         self.dropout = nn.Dropout(config.final_dropout)
#         self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

#     def forward(self, features, **kwargs):
#         x = features
#         x = self.dropout(x)
#         x = self.dense(x)
#         x = torch.tanh(x)
#         x = self.dropout(x)
#         x = self.out_proj(x)
#         return x


# class HubertForSpeechClassification(HubertPreTrainedModel):
#     def __init__(self, config):
#         super().__init__(config)
#         self.num_labels = config.num_labels
#         self.pooling_mode = config.pooling_mode
#         self.config = config

#         self.hubert = HubertModel(config)
#         self.classifier = HubertClassificationHead(config)

#         self.init_weights()

#     def freeze_feature_extractor(self):
#         self.hubert.feature_extractor._freeze_parameters()

#     def merged_strategy(
#             self,
#             hidden_states,
#             mode="mean"
#     ):
#         if mode == "mean":
#             outputs = torch.mean(hidden_states, dim=1)
#         elif mode == "sum":
#             outputs = torch.sum(hidden_states, dim=1)
#         elif mode == "max":
#             outputs = torch.max(hidden_states, dim=1)[0]
#         else:
#             raise Exception(
#                 "The pooling method hasn't been defined! Your pooling mode must be one of these ['mean', 'sum', 'max']")

#         return outputs

#     def forward(
#             self,
#             input_values,
#             attention_mask=None,
#             output_attentions=None,
#             output_hidden_states=None,
#             return_dict=None,
#             labels=None,
#     ):
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict
#         outputs = self.hubert(
#             input_values,
#             attention_mask=attention_mask,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#         )
#         hidden_states = outputs[0]
#         hidden_states = self.merged_strategy(hidden_states, mode=self.pooling_mode)
#         logits = self.classifier(hidden_states)

#         loss = None
#         if labels is not None:
#             if self.config.problem_type is None:
#                 if self.num_labels == 1:
#                     self.config.problem_type = "regression"
#                 elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
#                     self.config.problem_type = "single_label_classification"
#                 else:
#                     self.config.problem_type = "multi_label_classification"

#             if self.config.problem_type == "regression":
#                 loss_fct = MSELoss()
#                 loss = loss_fct(logits.view(-1, self.num_labels), labels)
#             elif self.config.problem_type == "single_label_classification":
#                 loss_fct = CrossEntropyLoss()
#                 loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
#             elif self.config.problem_type == "multi_label_classification":
#                 loss_fct = BCEWithLogitsLoss()
#                 loss = loss_fct(logits, labels)

#         if not return_dict:
#             output = (logits,) + outputs[2:]
#             return ((loss,) + output) if loss is not None else output

#         return SpeechClassifierOutput(
#             loss=loss,
#             logits=logits,
#             hidden_states=outputs.hidden_states,
#             attentions=outputs.attentions,
#         )

# def speech_file_to_array_fn(path, sampling_rate):
#     # speech_array, _sampling_rate = torchaudio.load(path)
#     # resampler = torchaudio.transforms.Resample(_sampling_rate)
#     # speech = resampler(speech_array).squeeze().numpy()
#     # return speech


#     speech, sr = sf.read(BytesIO(path))
#     speech = librosa.resample(np.asarray(speech), sr, sampling_rate)
#     #speech = speech[0].numpy().squeeze()
#     return speech

# def predict(path, sampling_rate):
#     speech = speech_file_to_array_fn(path, sampling_rate)
#     features = feature_extractor(speech, sampling_rate=sampling_rate, return_tensors="pt", padding=True)

#     input_values = features.input_values.to(device)

#     with torch.no_grad():
#         logits = model(input_values).logits

#     scores = F.softmax(logits, dim=1).detach().cpu().numpy()[0]
#     max_index = np.argmax(scores)
#     outputs = config.id2label[max_index]
#     return outputs

# config = AutoConfig.from_pretrained(
#     load_pretrainedmodel_path,
#     num_labels=num_labels,
#     label2id={label: i for i, label in enumerate(label_list)},
#     id2label={i: label for i, label in enumerate(label_list)},
#     finetuning_task="wav2vec2_clf",
# )
# setattr(config, 'pooling_mode', pooling_mode)

# feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(load_pretrainedprocessor_path)
# target_sampling_rate = feature_extractor.sampling_rate
# print(f"The target sampling rate: {target_sampling_rate}")

# model = HubertForSpeechClassification.from_pretrained(
#     load_pretrainedmodel_path,
#     config=config,
# )

# processor = feature_extractor.from_pretrained(load_pretrainedprocessor_path)
# model = model.from_pretrained(load_pretrainedmodel_path).to(device)

# class NLPService:
#     def __init__(self, model_dir:str):
#         '''
#         Parameters
#         ----------
#         model_dir : str
#             Path of model file to load.
#         '''
#         self.model_dir= model_dir
#         # TODO: Participant to complete.
#         pass

#     def locations_from_clues(self, clues:Iterable[Clue]) -> List[RealLocation]:
#         '''Process clues and get locations of interest.
        
#         Parameters
#         ----------
#         clues
#             Clues to process.

#         Returns
#         -------
#         lois
#             Locations of interest.
#         '''

#         # TODO: Participant to complete.
#         loi = []
#         for clue in clues:
#           pred = predict(clue[2], target_sampling_rate)
#           if pred == 'angry' or pred == 'sad':
#             loi.append(clue[1])
#             print (pred)
#         return loi 
#         #return [(3,1)]
#         pass

# class MockNLPService:
#     '''Mock NLP Service.
    
#     This is provided for testing purposes and should be replaced by your actual service implementation.
#     '''

#     def __init__(self, model_dir:str):
#         '''
#         Parameters
#         ----------
#         model_dir : str
#             Path of model file to load.
#         '''
#         self
#         pass

#     def locations_from_clues(self, clues:Iterable[Clue]) -> List[RealLocation]:
#         '''Process clues and get locations of interest.
        
#         Mock returns location of all clues.
#         '''
#         locations = [c.location for c in clues]

#         return locations



# WAV2VEC
from typing import Iterable, List
from tilsdk.localization.types import *
#import onnxruntime as ort
import numpy as np
import pandas as pd

from pathlib import Path
from tqdm import tqdm


import os
import sys

from transformers import AutoConfig, Wav2Vec2Processor

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, Wav2Vec2Processor
import soundfile as sf
from io import BytesIO
import librosa
import IPython.display as ipd
import numpy as np
import pandas as pd

from dataclasses import dataclass
from typing import Optional, Tuple
import torch
from transformers.file_utils import ModelOutput

from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2PreTrainedModel,
    Wav2Vec2Model
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
label_list = ['angry', 'fear', 'happy', 'neutral', 'sad']
num_labels = len(label_list)
# load_pretrainedmodel_path= f'{root_dir}/stubs/model/model_checkpoints4350+2epoch_huge2_smaller10+11'
# load_pretrainedprocessor_path = f'{root_dir}/stubs/model/processor_checkpoints4350+2epoch_huge2_smaller10+11'
load_pretrainedmodel_path= r'C:\Users\intern\Downloads\til-final-2022.1.1-best\til-final-2022.1.1-lower\stubs\model\model_checkpoints4350+2epoch_huge2_smaller10+11-20220618T165552Z-001'
load_pretrainedprocessor_path = r'C:\Users\intern\Downloads\til-final-2022.1.1-best\til-final-2022.1.1-lower\stubs\model\processor_checkpoints4350+2epoch_huge2_smaller10+11-20220618T165530Z-001'
pooling_mode = "mean"
is_regression = False


@dataclass
class SpeechClassifierOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None



class Wav2Vec2ClassificationHead(nn.Module):
    """Head for wav2vec classification task."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class Wav2Vec2ForSpeechClassification(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.pooling_mode = config.pooling_mode
        self.config = config

        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = Wav2Vec2ClassificationHead(config)

        self.init_weights()

    def freeze_feature_extractor(self):
        self.wav2vec2.feature_extractor._freeze_parameters()

    def merged_strategy(
            self,
            hidden_states,
            mode="mean"
    ):
        if mode == "mean":
            outputs = torch.mean(hidden_states, dim=1)
        elif mode == "sum":
            outputs = torch.sum(hidden_states, dim=1)
        elif mode == "max":
            outputs = torch.max(hidden_states, dim=1)[0]
        else:
            raise Exception(
                "The pooling method hasn't been defined! Your pooling mode must be one of these ['mean', 'sum', 'max']")

        return outputs

    def forward(
            self,
            input_values,
            attention_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            labels=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
        hidden_states = self.merged_strategy(hidden_states, mode=self.pooling_mode)
        logits = self.classifier(hidden_states)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SpeechClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

def speech_file_to_array_fn(path, sampling_rate):
    speech, sr = sf.read(BytesIO(path))
    speech = librosa.resample(np.asarray(speech), sr, sampling_rate)
    #speech = speech[0].numpy().squeeze()
    return speech

def predict(path, sampling_rate):
    speech = speech_file_to_array_fn(path, sampling_rate)
    features = processor(speech, sampling_rate=sampling_rate, return_tensors="pt", padding=True)

    input_values = features.input_values.to(device)
    attention_mask = features.attention_mask.to(device)

    with torch.no_grad():
        logits = model(input_values, attention_mask=attention_mask).logits

    scores = F.softmax(logits, dim=1).detach().cpu().numpy()[0]
    max_index = np.argmax(scores)
    outputs = config.id2label[max_index]
    return outputs


config = AutoConfig.from_pretrained(
    load_pretrainedmodel_path,
    num_labels=num_labels,
    label2id={label: i for i, label in enumerate(label_list)},
    id2label={i: label for i, label in enumerate(label_list)},
    finetuning_task="wav2vec2_clf",
)
setattr(config, 'pooling_mode', pooling_mode)

model = Wav2Vec2ForSpeechClassification.from_pretrained(
    load_pretrainedmodel_path,
    config=config,
)

model = model.from_pretrained(load_pretrainedmodel_path).to(device)
processor = Wav2Vec2Processor.from_pretrained(load_pretrainedprocessor_path )
target_sampling_rate = processor.feature_extractor.sampling_rate



class NLPService:
    def __init__(self, model_dir: str):
        '''
        Parameters
        ----------
        model_dir : str
            Path of model file to load.
        '''
        self.model_dir = model_dir
        # TODO: Participant to complete.
        pass

    def locations_from_clues(self, clues: Iterable[Clue]) -> List[RealLocation]:
        '''Process clues and get locations of interest.

        Parameters
        ----------
        clues
            Clues to process.

        Returns
        -------
        lois
            Locations of interest.
        '''

        # TODO: Participant to complete.
        loi = []
        for clue in clues:
            pred = predict(clue[2], target_sampling_rate)
            if pred == 'angry' or pred == 'sad':
                loi.append(clue[1])
                print(pred)
        return loi
        # return [(3,1)]
        pass


class MockNLPService:
    '''Mock NLP Service.

    This is provided for testing purposes and should be replaced by your actual service implementation.
    '''

    def __init__(self, model_dir: str):
        '''
        Parameters
        ----------
        model_dir : str
            Path of model file to load.
        '''
        self
        pass

    def locations_from_clues(self, clues: Iterable[Clue]) -> List[RealLocation]:
        '''Process clues and get locations of interest.

        Mock returns location of all clues.
        '''
        locations = [c.location for c in clues]

        return locations