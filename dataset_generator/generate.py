from benign import *
from developer_mode import *
from jailbreak import *
from prompt_leak import *
from roleplay import *
import random
import pandas as pd


dataset = []

#jailbreak

for template in  jailbreak_templates:
    for target in jailbreak_topics:
        dataset.append(template.format(target,0))


#benign

for template in benign_templates:
    for topic in benign_topics:
        dataset.append(template.format(topic,0))

#prompt leak

for template in promptleak_templates:
    for target in promptleak_targets:
        dataset.append(template.format(target,1))

#roleplay

for template in roleplay_templates:
    for topic in roleplay_topics:
        dataset.append(template.format(topic,0))


#developer_mode

for template in developer_templates:
    for action in developer_actions:
        dataset.append(template.format(action,1))


random.shuffle(dataset)

df = pd.DataFrame(
    dataset,
    columns=["text","label"]
)

df.to_csv(
    "../datasets/generated.csv",
    index=False
)