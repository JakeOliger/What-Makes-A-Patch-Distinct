# "What Makes a Patch Distinct?": An Attempt

### A Project by Jake Oliger

This project was the sole work of Jake Oliger. Huge thanks of course to Margolin et. al. who invented the algorithm used and wrote the paper on it, "What Makes a Patch Distinct?". A proper citation is given in `paper.pdf`.

The main report can be found in `final.pdf` -- this is merely a guide to running the program.

## Dependencies

For convenience, the pip package names are used so the dependencies can be easily installed using `pip install <package-name>`.

- `scikit-learn`
- `scikit-image`
- `matplotlib`
- `numpy`
- `Pillow`

## Running

Run the program using `python3 main.py`. It has been tested on Python 3.8.8 (using `python3` on Burrow) and 3.9 (on my local machine).

We have not added commandline arguments for specifying the image, so instead just change the value of `filepath` on line 220 of `main.py` to the desired image name. Do not add `images/` to the filepath -- the program will assume that is where it is located. The program will spit out weird directories or possibly crash if non-PNG, non-JPG images are used.

## Output

You can find the program output in `output/` in a directory that should match the image filename, minus the extension. For example, `shepherd.jpg`'s results will end up in `output/shepherd/`. The output will be two images: the raw salience as `salience.jpg` and the color-masked salience as `salience-color.jpg`.

There is also a trove of debug outputs that can be found with a similar structure in the directory `debug/`. Included are the outputs of most steps the program takes along the way, identified by a label of the step or as a type of salience, the scaling used, and the patch size used. The SLIC superpixel maps and their corresponding scales are also saved there. The same images sent to `output/` are also saved there for good measure.