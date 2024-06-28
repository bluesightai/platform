Welcome to Malevich! Here are simple steps to start building on our platform:

## Before anything

Before starting, you have to register on either [Space](https://space.malevich.ai/) or [Dev Space](https://dev.space.malevich.ai/). After that,
install a `malevich` package from development upstream using the following command:

```bash
pip install https://github.com/MalevichAI/malevich/archive/dev/unstable.zip
```

Then, initialize the project and login into your account:

```bash
malevich init && malevich space login
```

or if you are on dev space:

```bash
malevich space login --space-url https://dev.space.malevich.ai/
```

## Pushing model checkpoints

To push model checkpoints, use the following command

```bash
malevich core assets upload <PATH_ON_CLOUD> <LOCAL_PATH>
```

to check uploaded assets use:

```bash
malevich core assets list
```

Here is an example of pushing the asset and using in the app:

```file.txt

Hello from Malevich!
```

```

malevich core assets upload example_file.txt file.txt
malevich core assets list

>>>

assets 
└── example_file.txt

<<<
```

Creating a processor to just return the output of the file `example_file.txt`

```python

@processor()
def write_asset_file(just_input: DF, context: Context):
    with open(context.get_object('example_file.txt').path) as f:
        return pd.DataFrame([f.read()], columns=['contents'])

```

And then write a dummy flow just to test it. A flow should contain at least two nodes, 
so we pass a collection with dummy data to check the processor.

```python

@flow
def just_test_asset():
    return write_asset_file(
        collection('test', df=table(['some_data'], columns=['some_column']))) # just mock input to test

```

## Building and Using Apps

To run the app on the platform it should be built and pushed to Docker registry. You may use public and private Docker image registries
for this (GHCR, AWS, Dockerhub). Check more information at [Docs](https://docs.malevich.ai/SDK/Apps/Building.html#building-an-app). Shortly, I have built the image with:

```bash
(cd clay/ && docker build -t teexone/malevich:clay . && docker push teexone/malevich:clay)
```

and then installed it with 

```bash
malevich use image clay teexone/malevich:clay
```

After the installation, the app appears as 

```
from malevich.clay import ... # processors
```

