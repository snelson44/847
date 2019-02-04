# Neuromorphic Computing Lab 1

This code takes a list of 70000 images and applies different filters to them in order to determine spike positions and spike times, as well as analyze the relationship between the two

## Code base

`lab1.py` is where everything is ran. There are two other files, `firstlayer.py` and `filters.py`. These files contain classes that are utilized by `lab1.py`.

### Config

There is a `config.yaml` that is loaded into `lab1.py`. This is how I organized user inputs that may need to change. It is a more robust alternative to hardcoding.

## Running images

When `lab1.py` is ran, it automatically loads in all 70,000 images. In order to run analysis on all images, run the following command: 
```
lab1.py --run_all_images
```

In order to run only a select few (i.e. images 7 68 and 742):
```
lab1.py -i 7 68 742
```
If the -i flag is not used, only the first image in the list will be run.

## Filters

There are 3 filters. oncenter, offcenter, and sobel. The commands to run each filter are:

```
lab1.py -f oncenter
lab1.py -f offcenter
lab1.py -f sobel
```

## Outputs

Two plots will be output for each image, one of the spikepositions and one of the spiketimes.

The spikes inside the 3x3 receptive field of the final image will also be reported in a csv named `spiketimes.csv`

## Notes for grading

I would like to explain the considerable changes I made to the starter code here.

I created a seperate file/class for the filters. This way the code is more robust to easily adding more filters in the future. There are also several functions that where designed to be independent of their suuroundings. In other words, they are useful beyond just the scope of this singular lab. This adds up to more lines of code now, but ensures more stability and higher accesibility to editing in the future. All functions are clearly documented and explained.