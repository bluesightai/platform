import pickle

from loguru import logger

file_list = [
    "./data/0|48.478276213781456:-101.42501580501421|48.35933631386458:-101.22999959216078|naip|128|1.0|0.6.pkl",
    "./data/1|48.478276213781456:-101.42501580501421|48.35933631386458:-101.22999959216078|naip|128|1.0|0.6.pkl",
    "./data/2|48.478276213781456:-101.42501580501421|48.35933631386458:-101.22999959216078|naip|128|1.0|0.6.pkl",
    "./data/3|48.478276213781456:-101.42501580501421|48.35933631386458:-101.22999959216078|naip|128|1.0|0.6.pkl",
    "./data/4|48.478276213781456:-101.42501580501421|48.35933631386458:-101.22999959216078|naip|128|1.0|0.6.pkl",
    "./data/5|48.478276213781456:-101.42501580501421|48.35933631386458:-101.22999959216078|naip|128|1.0|0.6.pkl",
    "./data/6|48.478276213781456:-101.42501580501421|48.35933631386458:-101.22999959216078|naip|128|1.0|0.6.pkl",
]
filename_output = "./data/dakota_big.pkl"

data = []
for filename in file_list:
    with open(filename, "rb") as f:
        data_part = pickle.load(f)
    logger.info(f"Loaded {len(data_part)} items from {filename}!")
    data += data_part

logger.info(f"Got {len(data)} items in total!")

with open(filename_output, "wb") as f:
    pickle.dump(data, f)
logger.info(f"Dumped into {filename_output}")

# with open("london_manchester.pkl", 'wb') as f:
# pickle.dump(data, f)
